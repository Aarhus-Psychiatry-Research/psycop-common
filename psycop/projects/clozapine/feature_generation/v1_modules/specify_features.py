"""Feature specification module."""

import logging
from typing import Union

import numpy as np
from timeseriesflattener.v1.aggregation_fns import (
    AggregationFunType,
    boolean,
    count,
    earliest,
    maximum,
    summed,
    unique_count,
)
from timeseriesflattener.v1.feature_specs.group_specs import (
    NamedDataframe,
    OutcomeGroupSpec,
    PredictorGroupSpec,
)
from timeseriesflattener.v1.feature_specs.single_specs import (
    AnySpec,
    BaseModel,
    OutcomeSpec,
    PredictorSpec,
    StaticSpec,
)

from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.projects.clozapine.feature_generation.cohort_definition.clozapine_cohort_definition import (
    ClozapineCohortDefiner,
)
from psycop.projects.clozapine.feature_generation.cohort_definition.outcome_specification.first_clozapine_prescription import (
    get_first_clozapine_prescription,
)
from psycop.projects.clozapine.loaders.coercion import (
    beroligende_medicin,
    skema_1,
    skema_2_without_nutrition,
    skema_3,
)
from psycop.projects.clozapine.loaders.demographics import sex_female
from psycop.projects.clozapine.loaders.diagnoses import (
    cluster_b,
    f0_disorders,
    f1_disorders,
    f3_disorders,
    f4_disorders,
    f5_disorders,
    f6_disorders,
    f7_disorders,
    f8_disorders,
    f9_disorders_without_f99,
    manic_and_bipolar,
)
from psycop.projects.clozapine.loaders.lab_results import (
    cancelled_standard_lab_results,
    p_aripiprazol,
    p_clozapine,
    p_ethanol,
    p_haloperidol,
    p_olanzapine,
    p_paliperidone,
    p_paracetamol,
    p_risperidone,
)
from psycop.projects.clozapine.loaders.medications import (
    alcohol_abstinence,
    analgesic,
    antidepressives,
    antipsychotics,
    anxiolytics,
    aripiprazole_depot,
    first_gen_antipsychotics,
    haloperidol_depot,
    hyperactive_disorders_medications,
    hypnotics,
    lithium,
    olanzapine,
    olanzapine_depot,
    opioid_dependence,
    paliperidone_depot,
    perphenazine_depot,
    risperidone_depot,
    second_gen_antipsychotics,
    zuclopenthixol_depot,
)
from psycop.projects.clozapine.loaders.structured_sfi import (
    broeset_violence_checklist,
    suicide_risk_assessment,
)
from psycop.projects.clozapine.loaders.visits import (
    admissions_clozapine_2024,
    physical_visits_to_psychiatry_clozapine_2024,
    physical_visits_to_somatic_clozapine_2024,
)

log = logging.getLogger(__name__)


class SpecSet(BaseModel):
    """A set of unresolved specs, ready for resolving."""

    temporal_predictors: list[PredictorSpec]
    static_predictors: list[StaticSpec]
    outcomes: list[OutcomeSpec]
    metadata: list[AnySpec]


class FeatureSpecifier:
    def __init__(
        self,
        project_info: ProjectInfo,
        min_set_for_debug: bool = False,
        limited_feature_set: bool = False,
        unique_count_antipsychotics: bool = False,
    ):
        self.min_set_for_debug = min_set_for_debug
        self.project_info = project_info
        self.limited_feature_set = limited_feature_set
        self.unique_count_antipsychotics = unique_count_antipsychotics

    def _get_static_predictor_specs(self) -> list[StaticSpec]:
        """Get static predictor specs."""
        log.info("-------- Generating static specs --------")
        return [
            StaticSpec(
                timeseries_df=sex_female(),
                prefix=self.project_info.prefix.predictor,
                feature_base_name="sex_female",
            )
        ]

    def _get_outcome_specs(self) -> list[OutcomeSpec]:
        """Get outcome specs."""
        log.info("-------- Generating outcome specs --------")

        return OutcomeGroupSpec(
            named_dataframes=[
                NamedDataframe(
                    df=ClozapineCohortDefiner.get_outcome_timestamps().frame.to_pandas(),
                    name="first_clozapine_prescription",
                )
            ],
            lookahead_days=[year * 365 for year in (1, 2)],
            aggregation_fns=[maximum],
            fallback=[0],
            incident=[False],
            prefix=self.project_info.prefix.outcome,
        ).create_combinations()

    def _get_outcome_timestamp_specs(self) -> list[OutcomeSpec]:
        """Get outcome specs."""
        log.info("-------- Generating outcome specs --------")

        return OutcomeGroupSpec(
            named_dataframes=[
                NamedDataframe(
                    df=get_first_clozapine_prescription(timestamp_as_value_col=True), name=""
                )
            ],
            lookahead_days=[year * 365 for year in (1, 2)],
            aggregation_fns=[earliest],
            fallback=[np.NaN],
            incident=[False],
            prefix="timestamp_outcome",
        ).create_combinations()

    def _get_visits_specs(
        self, resolve_multiple: list[AggregationFunType], interval_days: list[float]
    ) -> list[PredictorSpec]:
        """Get visits specs."""
        log.info("-------- Generating visits specs --------")

        visits = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(
                    df=physical_visits_to_psychiatry_clozapine_2024(
                        return_value_as_visit_length_days=False
                    ),
                    name="physical_visits_to_psychiatry",
                ),
                NamedDataframe(
                    df=physical_visits_to_somatic_clozapine_2024(),
                    name="physical_visits_to_somatic",
                ),
            ),
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[0],
        ).create_combinations()

        return visits

    def _get_admissions_specs(
        self, resolve_multiple: list[AggregationFunType], interval_days: list[float]
    ) -> list[PredictorSpec]:
        """Get admissions specs."""
        log.info("-------- Generating admissions specs --------")

        admissions_df = PredictorGroupSpec(
            named_dataframes=[
                NamedDataframe(
                    df=admissions_clozapine_2024(return_value_as_visit_length_days=True),
                    name="admissions",
                )
            ],
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[0],
        ).create_combinations()

        return admissions_df

    def _get_medication_specs(
        self, resolve_multiple: list[AggregationFunType], interval_days: list[float]
    ) -> list[PredictorSpec]:
        """Get medication specs."""
        log.info("-------- Generating medication specs --------")

        psychiatric_medications = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=antipsychotics(), name="antipsychotics"),
                NamedDataframe(df=first_gen_antipsychotics(), name="first_gen_antipsychotics"),
                NamedDataframe(df=second_gen_antipsychotics(), name="second_gen_antipsychotics"),
                NamedDataframe(df=olanzapine(), name="olanzapine"),
                NamedDataframe(df=anxiolytics(), name="anxiolytics"),
                NamedDataframe(df=hypnotics(), name="hypnotics and sedatives"),
                NamedDataframe(df=antidepressives(), name="antidepressives"),
                NamedDataframe(df=lithium(), name="lithium"),
                NamedDataframe(
                    df=hyperactive_disorders_medications(), name="hyperactive disorders medications"
                ),
                NamedDataframe(df=alcohol_abstinence(), name="alcohol_abstinence"),
                NamedDataframe(df=opioid_dependence(), name="opioid_dependence"),
                NamedDataframe(df=analgesic(), name="analgesics"),
                NamedDataframe(df=olanzapine_depot(), name="olanzapine_depot"),
                NamedDataframe(df=aripiprazole_depot(), name="aripiprazole_depot"),
                NamedDataframe(df=risperidone_depot(), name="risperidone_depot"),
                NamedDataframe(df=paliperidone_depot(), name="paliperidone_depot"),
                NamedDataframe(df=haloperidol_depot(), name="haloperidol_depot"),
                NamedDataframe(df=perphenazine_depot(), name="perphenazine_depot"),
                NamedDataframe(df=zuclopenthixol_depot(), name="zuclopenthixol_depot"),
            ),
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[0],
        ).create_combinations()

        return psychiatric_medications

    def _get_unique_count_medication_specs(
        self, resolve_multiple: list[AggregationFunType], interval_days: list[float]
    ) -> list[PredictorSpec]:
        """Get unique count medications."""
        log.info("-------- Generating unique count medications specs --------")

        unique_count_medications = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(
                    df=antipsychotics(add_code_to_output_col=True), name="antipsychotics"
                ),
                NamedDataframe(
                    df=antidepressives(add_code_to_output_col=True), name="antidepressives"
                ),
                NamedDataframe(df=anxiolytics(add_code_to_output_col=True), name="anxiolytics"),
                NamedDataframe(df=analgesic(add_code_to_output_col=True), name="analgesics"),
            ),
            aggregation_fns=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[0],
        ).create_combinations()

        return unique_count_medications

    def _get_diagnoses_specs(
        self, resolve_multiple: list[AggregationFunType], interval_days: list[float]
    ) -> list[PredictorSpec]:
        """Get diagnoses specs."""
        log.info("-------- Generating diagnoses specs --------")

        psychiatric_diagnoses = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=f0_disorders(), name="f0_disorders"),
                NamedDataframe(df=f1_disorders(), name="f1_disorders"),
                NamedDataframe(df=f3_disorders(), name="f3_disorders"),
                NamedDataframe(df=manic_and_bipolar(), name="manic_and_bipolar"),
                NamedDataframe(df=f4_disorders(), name="f4_disorders"),
                NamedDataframe(df=f5_disorders(), name="f5_disorders"),
                NamedDataframe(df=f6_disorders(), name="f6_disorders"),
                NamedDataframe(df=cluster_b(), name="cluster_b"),
                NamedDataframe(df=f7_disorders(), name="f7_disorders"),
                NamedDataframe(df=f8_disorders(), name="f8_disorders"),
                NamedDataframe(df=f9_disorders_without_f99(), name="f9_disorders"),
            ),
            aggregation_fns=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[0],
        ).create_combinations()

        return psychiatric_diagnoses

    def _get_coercion_specs(
        self, resolve_multiple: list[AggregationFunType], interval_days: list[float]
    ) -> list[PredictorSpec]:
        """Get coercion specs."""
        log.info("-------- Generating coercion specs --------")

        coercion = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=skema_1(), name="skema_1"),
                NamedDataframe(df=skema_2_without_nutrition(), name="skema_2_without_nutrition"),
                NamedDataframe(df=skema_3(), name="skema_3"),
            ),
            aggregation_fns=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[0],
        ).create_combinations()

        return coercion

    def _get_beroligende_medicin_specs(
        self, resolve_multiple: list[AggregationFunType], interval_days: list[float]
    ) -> list[PredictorSpec]:
        """Get beroligende medicin specs."""
        log.info("-------- Generating beroligende medicicn specs --------")

        beroligende_medicin_df = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=beroligende_medicin(), name="beroligende_medicin"),
            ),
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[0],
        ).create_combinations()

        return beroligende_medicin_df

    def _get_structured_sfi_specs(
        self, resolve_multiple: list[AggregationFunType], interval_days: list[float]
    ) -> list[PredictorSpec]:
        """Get structured sfi specs."""
        log.info("-------- Generating structured sfi specs --------")

        structured_sfi = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=broeset_violence_checklist(), name="broeset_violence_checklist"),
                NamedDataframe(df=suicide_risk_assessment(), name="selvmordsrisiko"),
            ),
            aggregation_fns=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[np.nan],
        ).create_combinations()

        return structured_sfi

    def _get_lab_result_specs(
        self, resolve_multiple: list[AggregationFunType], interval_days: list[float]
    ) -> list[PredictorSpec]:
        """Get lab result specs."""
        log.info("-------- Generating lab result specs --------")

        lab_results = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=p_clozapine(), name="p_clozapine"),
                NamedDataframe(df=p_olanzapine(), name="p_olanzapine"),
                NamedDataframe(df=p_aripiprazol(), name="p_aripiprazol"),
                NamedDataframe(df=p_risperidone(), name="p_risperidone"),
                NamedDataframe(df=p_paliperidone(), name="p_paliperidone"),
                NamedDataframe(df=p_haloperidol(), name="p_haloperidol"),
                NamedDataframe(df=p_paracetamol(), name="p_paracetamol"),
                NamedDataframe(df=p_ethanol(), name="p_ethanol"),
            ),
            aggregation_fns=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[np.nan],
        ).create_combinations()

        return lab_results

    def _get_cancelled_lab_result_specs(
        self, resolve_multiple: list[AggregationFunType], interval_days: list[float]
    ) -> list[PredictorSpec]:
        """Get cancelled lab result specs."""
        log.info("-------- Generating cancelled lab result specs --------")

        cancelled_lab_results = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(
                    df=cancelled_standard_lab_results(), name="cancelled_standard_lab_results"
                ),
            ),
            aggregation_fns=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[0],
        ).create_combinations()

        return cancelled_lab_results

    def _get_limited_feature_specs(self) -> list[PredictorSpec]:
        """Get limited feature set specs."""
        log.info("-------- Generating limited feature set specs --------")

        limited_feature_set = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=f3_disorders(), name="f3_disorders"),
                NamedDataframe(df=olanzapine(), name="olanzapine"),
            ),
            aggregation_fns=[boolean],
            lookbehind_days=[365],
            fallback=[np.nan],
        ).create_combinations()

        return limited_feature_set

    def _get_temporal_predictor_specs(self) -> list[PredictorSpec]:
        """Generate predictor spec list."""
        log.info("-------- Generating temporal predictor specs --------")

        if self.min_set_for_debug:
            return [
                PredictorSpec(
                    timeseries_df=antipsychotics(),
                    feature_base_name="antipsychotics",
                    lookbehind_days=30,
                    aggregation_fn=maximum,
                    fallback=np.nan,
                    prefix=self.project_info.prefix.predictor,
                )
            ]

        if self.unique_count_antipsychotics:
            return [
                PredictorSpec(
                    timeseries_df=antipsychotics(add_code_to_output_col=True),
                    feature_base_name="antipsychotics",
                    lookbehind_days=365,
                    aggregation_fn=unique_count,
                    fallback=np.nan,
                    prefix=self.project_info.prefix.predictor,
                )
            ]

        interval_days = [30.0, 180.0, 365.0]

        visits = self._get_visits_specs(resolve_multiple=[count], interval_days=interval_days)

        admissions = self._get_admissions_specs(
            resolve_multiple=[count, summed], interval_days=interval_days
        )

        diagnoses = self._get_diagnoses_specs(
            resolve_multiple=[count, boolean, unique_count], interval_days=interval_days
        )

        medications = self._get_medication_specs(
            resolve_multiple=[count, boolean], interval_days=interval_days
        )

        unique_count_medications = self._get_unique_count_medication_specs(
            resolve_multiple=[unique_count], interval_days=[365.0]
        )

        beroligende_medicin = self._get_beroligende_medicin_specs(
            resolve_multiple=[count, boolean], interval_days=interval_days
        )

        lab_results = self._get_lab_result_specs(
            resolve_multiple=[summed, boolean], interval_days=interval_days
        )

        coercion = self._get_coercion_specs(
            resolve_multiple=[count, summed, boolean], interval_days=interval_days
        )
        structured_sfi = self._get_structured_sfi_specs(
            resolve_multiple=[summed, maximum], interval_days=interval_days
        )

        cancelled_lab_results = self._get_cancelled_lab_result_specs(
            resolve_multiple=[count, boolean], interval_days=interval_days
        )

        return (
            visits
            + admissions
            + medications
            + unique_count_medications
            + diagnoses
            + beroligende_medicin
            + lab_results
            + cancelled_lab_results
            + coercion
            + structured_sfi
        )

    def get_feature_specs(self) -> list[Union[StaticSpec, OutcomeSpec, PredictorSpec]]:
        """Get a spec set."""

        if self.min_set_for_debug:
            log.warning("--- !!! Using the minimum set of features for debugging !!! ---")
            return (
                self._get_temporal_predictor_specs()
                + self._get_static_predictor_specs()
                + self._get_outcome_specs()
                + self._get_outcome_timestamp_specs()
            )
        if self.limited_feature_set:
            return (
                self._get_limited_feature_specs()
                + self._get_static_predictor_specs()
                + self._get_outcome_specs()
                + self._get_outcome_timestamp_specs()
            )

        if self.unique_count_antipsychotics:
            return (
                self._get_temporal_predictor_specs()
                + self._get_static_predictor_specs()
                + self._get_outcome_specs()
                + self._get_outcome_timestamp_specs()
            )

        return (
            self._get_temporal_predictor_specs()
            + self._get_static_predictor_specs()
            + self._get_outcome_specs()
            + self._get_outcome_timestamp_specs()
        )
