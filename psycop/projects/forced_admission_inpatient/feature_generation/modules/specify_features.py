"""Feature specification module."""
import logging
from typing import Callable, Union

import numpy as np
from timeseriesflattener.aggregation_fns import (
    boolean,
    change_per_day,
    count,
    latest,
    maximum,
    mean,
    variance,
)
from timeseriesflattener.feature_specs.group_specs import (
    NamedDataframe,
    PredictorGroupSpec,
)
from timeseriesflattener.feature_specs.single_specs import (
    AnySpec,
    BaseModel,
    OutcomeSpec,
    PredictorSpec,
    StaticSpec,
)

from psycop.common.feature_generation.application_modules.project_setup import (
    ProjectInfo,
)
from psycop.common.feature_generation.loaders.raw.load_coercion import (
    af_legemlig_lidelse,
    baelte,
    beroligende_medicin,
    ect,
    farlighed,
    fastholden,
    medicinering,
    remme,
    skema_1,
    skema_2_without_nutrition,
    skema_3,
    tvangstilbageholdelse,
)
from psycop.common.feature_generation.loaders.raw.load_demographic import sex_female
from psycop.common.feature_generation.loaders.raw.load_diagnoses import (
    cluster_b,
    f0_disorders,
    f1_disorders,
    f2_disorders,
    f3_disorders,
    f4_disorders,
    f5_disorders,
    f6_disorders,
    f7_disorders,
    f8_disorders,
    f9_disorders,
    manic_and_bipolar,
    schizoaffective,
    schizophrenia,
)
from psycop.common.feature_generation.loaders.raw.load_lab_results import (
    cancelled_standard_lab_results,
    p_aripiprazol,
    p_clomipramine,
    p_clozapine,
    p_ethanol,
    p_haloperidol,
    p_lithium,
    p_nortriptyline,
    p_olanzapine,
    p_paliperidone,
    p_paracetamol,
    p_risperidone,
)
from psycop.common.feature_generation.loaders.raw.load_medications import (
    alcohol_abstinence,
    analgesic,
    antidepressives,
    antipsychotics,
    anxiolytics,
    aripiprazole_depot,
    clozapine,
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
from psycop.common.feature_generation.loaders.raw.load_structured_sfi import (
    broeset_violence_checklist,
    hamilton_d17,
    mas_m,
    selvmordsrisiko,
)
from psycop.common.feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
    physical_visits_to_somatic,
)

log = logging.getLogger(__name__)

from psycop.common.feature_generation.loaders.raw.load_visits import admissions


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
    ):
        self.min_set_for_debug = min_set_for_debug
        self.project_info = project_info
        self.limited_feature_set = limited_feature_set

    def _get_static_predictor_specs(self) -> list[StaticSpec]:
        """Get static predictor specs."""
        return [
            StaticSpec(
                timeseries_df=sex_female(),
                prefix=self.project_info.prefix.predictor,
                feature_base_name="sex_female",
            ),
        ]

    def _get_visits_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get visits specs."""
        log.info("-------- Generating visits specs --------")

        visits = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(
                    df=physical_visits_to_psychiatry(),
                    name="physical_visits_to_psychiatry",
                ),
                NamedDataframe(
                    df=physical_visits_to_somatic(),
                    name="physical_visits_to_somatic",
                ),
            ),
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[0],
        ).create_combinations()

        return visits

    def _get_admissions_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get admissions specs."""
        log.info("-------- Generating admissions specs --------")

        admissions_df = PredictorGroupSpec(
            named_dataframes=[NamedDataframe(df=admissions(), name="admissions")],
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[0],
        ).create_combinations()

        return admissions_df

    def _get_medication_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get medication specs."""
        log.info("-------- Generating medication specs --------")

        psychiatric_medications = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=antipsychotics(), name="antipsychotics"),
                NamedDataframe(
                    df=first_gen_antipsychotics(),
                    name="first_gen_antipsychotics",
                ),
                NamedDataframe(
                    df=second_gen_antipsychotics(),
                    name="second_gen_antipsychotics",
                ),
                NamedDataframe(df=olanzapine(), name="olanzapine"),
                NamedDataframe(df=clozapine(), name="clozapine"),
                NamedDataframe(df=anxiolytics(), name="anxiolytics"),
                NamedDataframe(df=hypnotics(), name="hypnotics and sedatives"),
                NamedDataframe(df=antidepressives(), name="antidepressives"),
                NamedDataframe(df=lithium(), name="lithium"),
                NamedDataframe(
                    df=hyperactive_disorders_medications(),
                    name="hyperactive disorders medications",
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

    def _get_diagnoses_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get diagnoses specs."""
        log.info("-------- Generating diagnoses specs --------")

        psychiatric_diagnoses = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=f0_disorders(), name="f0_disorders"),
                NamedDataframe(df=f1_disorders(), name="f1_disorders"),
                NamedDataframe(df=f2_disorders(), name="f2_disorders"),
                NamedDataframe(df=schizophrenia(), name="schizophrenia"),
                NamedDataframe(df=schizoaffective(), name="schizoaffective"),
                NamedDataframe(df=f3_disorders(), name="f3_disorders"),
                NamedDataframe(df=manic_and_bipolar(), name="manic_and_bipolar"),
                NamedDataframe(df=f4_disorders(), name="f4_disorders"),
                NamedDataframe(df=f5_disorders(), name="f5_disorders"),
                NamedDataframe(df=f6_disorders(), name="f6_disorders"),
                NamedDataframe(df=cluster_b(), name="cluster_b"),
                NamedDataframe(df=f7_disorders(), name="f7_disorders"),
                NamedDataframe(df=f8_disorders(), name="f8_disorders"),
                NamedDataframe(df=f9_disorders(), name="f9_disorders"),
            ),
            aggregation_fns=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[0],
        ).create_combinations()

        return psychiatric_diagnoses

    def _get_coercion_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get coercion specs."""
        log.info("-------- Generating coercion specs --------")

        coercion = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=skema_1(), name="skema_1"),
                NamedDataframe(
                    df=tvangstilbageholdelse(),
                    name="tvangstilbageholdelse",
                ),
                NamedDataframe(
                    df=skema_2_without_nutrition(),
                    name="skema_2_without_nutrition",
                ),
                NamedDataframe(df=medicinering(), name="medicinering"),
                NamedDataframe(df=ect(), name="ect"),
                NamedDataframe(df=af_legemlig_lidelse(), name="af_legemlig_lidelse"),
                NamedDataframe(df=skema_3(), name="skema_3"),
                NamedDataframe(df=fastholden(), name="fastholden"),
                NamedDataframe(df=baelte(), name="baelte"),
                NamedDataframe(df=remme(), name="remme"),
                NamedDataframe(df=farlighed(), name="farlighed"),
            ),
            aggregation_fns=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[0],
        ).create_combinations()

        return coercion

    def _get_beroligende_medicin_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
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
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get structured sfi specs."""
        log.info("-------- Generating structured sfi specs --------")

        structured_sfi = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(
                    df=broeset_violence_checklist(),
                    name="broeset_violence_checklist",
                ),
                NamedDataframe(df=selvmordsrisiko(), name="selvmordsrisiko"),
                NamedDataframe(df=hamilton_d17(), name="hamilton_d17"),
                NamedDataframe(df=mas_m(), name="mas_m"),
            ),
            aggregation_fns=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[np.nan],
        ).create_combinations()

        return structured_sfi

    def _get_lab_result_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get lab result specs."""
        log.info("-------- Generating lab result specs --------")

        lab_results = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=p_lithium(), name="p_lithium"),
                NamedDataframe(df=p_clozapine(), name="p_clozapine"),
                NamedDataframe(df=p_olanzapine(), name="p_olanzapine"),
                NamedDataframe(df=p_aripiprazol(), name="p_aripiprazol"),
                NamedDataframe(df=p_risperidone(), name="p_risperidone"),
                NamedDataframe(df=p_paliperidone(), name="p_paliperidone"),
                NamedDataframe(df=p_haloperidol(), name="p_haloperidol"),
                NamedDataframe(df=p_paracetamol(), name="p_paracetamol"),
                NamedDataframe(df=p_ethanol(), name="p_ethanol"),
                NamedDataframe(df=p_nortriptyline(), name="p_nortriptyline"),
                NamedDataframe(df=p_clomipramine(), name="p_clomipramine"),
                NamedDataframe(
                    df=cancelled_standard_lab_results(),
                    name="cancelled_standard_lab_results",
                ),
            ),
            aggregation_fns=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[np.nan],
        ).create_combinations()

        return lab_results

    def _get_limited_feature_specs(
        self,
    ) -> list[PredictorSpec]:
        """Get lab result specs."""
        log.info("-------- Generating limited feature set specs --------")

        limited_feature_set = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=f0_disorders(), name="f0_disorders"),
                NamedDataframe(df=f1_disorders(), name="f1_disorders"),
                NamedDataframe(df=f2_disorders(), name="f2_disorders"),
                NamedDataframe(df=f3_disorders(), name="f3_disorders"),
                NamedDataframe(df=f4_disorders(), name="f4_disorders"),
                NamedDataframe(df=f5_disorders(), name="f5_disorders"),
                NamedDataframe(df=f6_disorders(), name="f6_disorders"),
                NamedDataframe(df=f7_disorders(), name="f07_disorders"),
                NamedDataframe(df=f8_disorders(), name="f8_disorders"),
                NamedDataframe(df=f9_disorders(), name="f9_disorders"),
                NamedDataframe(df=skema_1(), name="skema_1"),
                NamedDataframe(
                    df=tvangstilbageholdelse(),
                    name="tvangstilbageholdelse",
                ),
                NamedDataframe(
                    df=skema_2_without_nutrition(),
                    name="skema_2_without_nutrition",
                ),
                NamedDataframe(df=medicinering(), name="medicinering"),
                NamedDataframe(df=ect(), name="ect"),
                NamedDataframe(df=af_legemlig_lidelse(), name="af_legemlig_lidelse"),
                NamedDataframe(df=skema_3(), name="skema_3"),
                NamedDataframe(df=fastholden(), name="fastholden"),
                NamedDataframe(df=baelte(), name="baelte"),
                NamedDataframe(df=remme(), name="remme"),
                NamedDataframe(df=farlighed(), name="farlighed"),
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
                    timeseries_df=schizophrenia(),
                    feature_base_name="schizophrenia",
                    lookbehind_days=30,
                    aggregation_fn=maximum,
                    fallback=np.nan,
                    prefix=self.project_info.prefix.predictor,
                ),
            ]

        interval_days = [10.0, 30.0, 180.0, 365.0]

        visits = self._get_visits_specs(
            resolve_multiple=[count],
            interval_days=interval_days,
        )

        admissions = self._get_admissions_specs(
            resolve_multiple=[count, sum],
            interval_days=interval_days,
        )

        diagnoses = self._get_diagnoses_specs(
            resolve_multiple=[count, boolean],
            interval_days=interval_days,
        )

        medications = self._get_medication_specs(
            resolve_multiple=[count, boolean],
            interval_days=interval_days,
        )

        beroligende_medicin = self._get_beroligende_medicin_specs(
            resolve_multiple=[count, boolean],
            interval_days=interval_days,
        )

        coercion = self._get_coercion_specs(
            resolve_multiple=[count, sum, boolean],
            interval_days=interval_days,
        )

        structured_sfi = self._get_structured_sfi_specs(
            resolve_multiple=[mean, max, min, change_per_day, variance],
            interval_days=interval_days,
        )

        lab_results = self._get_lab_result_specs(
            resolve_multiple=[max, min, mean, latest],
            interval_days=interval_days,
        )

        return (
            visits
            + admissions
            + medications
            + diagnoses
            + beroligende_medicin
            + coercion
            + structured_sfi
            + lab_results
        )

    def get_feature_specs(self) -> list[Union[StaticSpec, PredictorSpec]]:
        """Get a spec set."""

        if self.min_set_for_debug:
            return (
                self._get_temporal_predictor_specs()
                + self._get_static_predictor_specs()
            )
        if self.limited_feature_set:
            return (
                self._get_limited_feature_specs() + self._get_static_predictor_specs()
            )

        return self._get_temporal_predictor_specs() + self._get_static_predictor_specs()
