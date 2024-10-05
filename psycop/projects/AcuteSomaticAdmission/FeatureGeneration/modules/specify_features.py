"""Feature specification module."""
import logging
from typing import Union

import numpy as np
from timeseriesflattener.v1.aggregation_fns import (
    AggregationFunType,
    boolean,
    change_per_day,
    count,
    earliest,
    latest,
    maximum,
    mean,
    minimum,
    summed,
    variance,
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
    hba1c,
    scheduled_glc,
    unscheduled_p_glc,
    fasting_p_glc,
    triglycerides,
    fasting_triglycerides,
    hdl,
    ldl,
    fasting_ldl,
    alat,
    asat,
    lymphocytes,
    leukocytes,
    crp,
    creatinine,
    egfr,
    albumine_creatinine_ratio,
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
    top_10_weight_gaining_antipsychotics,
    sedative_antipsychotics,
    non_sedative_antipsychotics,
    benzodiazepines,
    benzodiazepine_related_sleeping_agents,
    pregabaline,
    hypnotics_and_rivotril,
    mood_stabilisers,
    nervous_system_medications,
    ssri,
    snri,
    tca,
    selected_nassa,
    valproate,
    lamotrigine,
    dementia_medications,
    anti_epileptics,
    alimentary_medications,
    blood_medications,
    cardiovascular_medications,
    dermatological_medications,
    genito_sex_medications,
    hormonal_medications,
    antiinfectives,
    antineoplastic,
    musculoskeletal_medications,
    antiparasitic,
    respiratory_medications,
    sensory_medications,
    various_medications,
    diuretics,
    statins,
    antihypertensives,
    diuretics,
    gerd_drugs,
)
from psycop.common.feature_generation.loaders.raw.load_structured_sfi import (
    broeset_violence_checklist,
    hamilton_d17,
    mas_m,
    selvmordsrisiko,
    height_in_cm,
    weight_in_kg,
    bmi,
    no_temporary_leave,
    temporary_leave,
    supervised_temporary_leave,
    unsupervised_temporary_leave,
    systolic_blood_pressure,
    diastolic_blood_pressure,
    pulse,
    smoking_continuous,
    smoking_categorical,
)
from psycop.common.feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
    physical_visits_to_somatic,
)
from psycop.projects.AcuteSomaticAdmission.CohortDefinition.get_somatic_emergency_visits import (
    get_contacts_to_somatic_emergency,
)

from psycop.common.feature_generation.loaders.raw.load_visits import admissions

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
        lookbehind_180d_mean: bool = False,
    ):
        self.min_set_for_debug = min_set_for_debug
        self.project_info = project_info
        self.limited_feature_set = limited_feature_set
        self.lookbehind_180d_mean = lookbehind_180d_mean

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
                NamedDataframe(df=get_contacts_to_somatic_emergency(), name="AcuteAdmission")
            ],
            lookahead_days=[30, 60, 90],
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
                    df=get_contacts_to_somatic_emergency(timestamp_as_value_col=True), name=""
                )
            ],
            lookahead_days=[30, 60, 90],
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
                    df=physical_visits_to_psychiatry(return_value_as_visit_length_days=False),
                    name="physical_visits_to_psychiatry",
                ),
                NamedDataframe(df=physical_visits_to_somatic(), name="physical_visits_to_somatic"),
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
                    df=admissions(return_value_as_visit_length_days=True), name="admissions"
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
                NamedDataframe(df=clozapine(), name="clozapine"),
                NamedDataframe(df=anxiolytics(), name="anxiolytics"),
                NamedDataframe(df=hypnotics(), name="hypnotics and sedatives"),
                NamedDataframe(df=antidepressives(), name="antidepressives"),
                NamedDataframe(df=lithium(), name="lithium"),
                NamedDataframe(df=hyperactive_disorders_medications(), name="hyperactive disorders medications"),
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
                NamedDataframe(df=top_10_weight_gaining_antipsychotics(), name="top_10_weight_gaining_antipsychotics"),
                NamedDataframe(df=sedative_antipsychotics(), name="sedative_antipsychotics"),
                NamedDataframe(df=non_sedative_antipsychotics(), name="non_sedative_antipsychotics"),
                NamedDataframe(df=benzodiazepines(), name="benzodiazepines"),
                NamedDataframe(df=benzodiazepine_related_sleeping_agents(), name="benzodiazepine_related_sleeping_agents"),
                NamedDataframe(df=pregabaline(), name="pregabaline"),
                NamedDataframe(df=hypnotics_and_rivotril(), name="hypnotics_and_rivotril"),
                NamedDataframe(df=mood_stabilisers(), name="mood_stabilisers"),
                NamedDataframe(df=nervous_system_medications(), name="nervous_system_medications"),
                NamedDataframe(df=ssri(), name="ssri"),
                NamedDataframe(df=snri(), name="snri"),
                NamedDataframe(df=tca(), name="tca"),
                NamedDataframe(df=selected_nassa(), name="selected_nassa"),
                NamedDataframe(df=valproate(), name="valproate"),
                NamedDataframe(df=lamotrigine(), name="lamotrigine"),
                NamedDataframe(df=dementia_medications(), name="dementia_medications"),
                 NamedDataframe(df=anti_epileptics(), name="anti_epileptics"),
                NamedDataframe(df=alimentary_medications(), name="alimentary_medications"),
                NamedDataframe(df=blood_medications(), name="blood_medications"),
                NamedDataframe(df=cardiovascular_medications(), name="cardiovascular_medications"),
                NamedDataframe(df=dermatological_medications(), name="dermatological_medications"),
                NamedDataframe(df=genito_sex_medications(), name="genito_sex_medications"),
                NamedDataframe(df=hormonal_medications(), name="hormonal_medications"),
                NamedDataframe(df=antiinfectives(), name="antiinfectives"),
                NamedDataframe(df=antineoplastic(), name="antineoplastic"),
                NamedDataframe(df=musculoskeletal_medications(), name="musculoskeletal_medications"),
                NamedDataframe(df=antiparasitic(), name="antiparasitic"),
                NamedDataframe(df=respiratory_medications(), name="respiratory_medications"),
                NamedDataframe(df=sensory_medications(), name="sensory_medications"),
                NamedDataframe(df=various_medications(), name="various_medications"),
                NamedDataframe(df=diuretics(), name="diuretics"),
            ),
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[0],
        ).create_combinations()

        return psychiatric_medications

    def _get_diagnoses_specs(
        self, resolve_multiple: list[AggregationFunType], interval_days: list[float]
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
        self, resolve_multiple: list[AggregationFunType], interval_days: list[float]
    ) -> list[PredictorSpec]:
        """Get coercion specs."""
        log.info("-------- Generating coercion specs --------")

        coercion = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=skema_1(), name="skema_1"),
                NamedDataframe(df=tvangstilbageholdelse(), name="tvangstilbageholdelse"),
                NamedDataframe(df=skema_2_without_nutrition(), name="skema_2_without_nutrition"),
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
                NamedDataframe(df=selvmordsrisiko(), name="selvmordsrisiko"),
                NamedDataframe(df=hamilton_d17(), name="hamilton_d17"),
                NamedDataframe(df=height_in_cm(), name="height_in_cm"),
                NamedDataframe(df=weight_in_kg(), name="weight_in_kg"),
                NamedDataframe(df=bmi(), name="bmi"),
                NamedDataframe(df=no_temporary_leave(), name="no_temporary_leave"),
                NamedDataframe(df=temporary_leave(), name="temporary_leave"),
                NamedDataframe(df=supervised_temporary_leave(), name="supervised_temporary_leave"),
                NamedDataframe(df=unsupervised_temporary_leave(), name="unsupervised_temporary_leave"),
                NamedDataframe(df=systolic_blood_pressure(), name="systolic_blood_pressure"),
                NamedDataframe(df=smoking_continuous(), name="smoking_continuous"),
                NamedDataframe(df=smoking_categorical(), name="smoking_categorical"),
                NamedDataframe(df=diastolic_blood_pressure(), name="siastolic_blood_pressure"),
                NamedDataframe(df=pulse(), name="pulse"),
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
                NamedDataframe(df=hba1c(), name="hba1c"),
                NamedDataframe(df=scheduled_glc(), name="scheduled_glc"),
                NamedDataframe(df=unscheduled_p_glc(), name="unscheduled_p_glc"),
                NamedDataframe(df=fasting_p_glc(), name="fasting_p_glc"),
                NamedDataframe(df=triglycerides(), name="triglycerides"),
                NamedDataframe(df=fasting_triglycerides(), name="fasting_triglycerides"),
                NamedDataframe(df=hdl(), name="hdl"),
                NamedDataframe(df=ldl(), name="ldl"),
                NamedDataframe(df=fasting_ldl(), name="fasting_ldl"),
                NamedDataframe(df=alat(), name="alat"),
                NamedDataframe(df=asat(), name="asat"),
                NamedDataframe(df=lymphocytes(), name="lymphocytes"),
                NamedDataframe(df=leukocytes(), name="leukocytes"),
                NamedDataframe(df=crp(), name="crp"),
                #NamedDataframe(df=creatinine(), name="creatinine"), TO DO: VIRKER IKKE
                NamedDataframe(df=egfr(), name="egfr"),
                NamedDataframe(df=albumine_creatinine_ratio(), name="albumine_creatinine_ratio"),
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
                NamedDataframe(df=f7_disorders(), name="f7_disorders"),
                NamedDataframe(df=f8_disorders(), name="f8_disorders"),
                NamedDataframe(df=f9_disorders(), name="f9_disorders"),
            ),
            aggregation_fns=[boolean],
            lookbehind_days=[180],
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
                )
            ]
        if self.lookbehind_180d_mean:
            interval_days = [180.0]
            resolve_multiple = [mean]

            visits = self._get_visits_specs(
                resolve_multiple=resolve_multiple, interval_days=interval_days
            )

            admissions = self._get_admissions_specs(
                resolve_multiple=resolve_multiple, interval_days=interval_days
            )

            diagnoses = self._get_diagnoses_specs(
                resolve_multiple=resolve_multiple, interval_days=interval_days
            )

            medications = self._get_medication_specs(
                resolve_multiple=resolve_multiple, interval_days=interval_days
            )

            beroligende_medicin = self._get_beroligende_medicin_specs(
                resolve_multiple=[boolean], interval_days=interval_days
            )

            coercion = self._get_coercion_specs(
                resolve_multiple=resolve_multiple, interval_days=interval_days
            )

            structured_sfi = self._get_structured_sfi_specs(
                resolve_multiple=resolve_multiple, interval_days=interval_days
            )

            lab_results = self._get_lab_result_specs(
                resolve_multiple=resolve_multiple, interval_days=interval_days
            )

            cancelled_lab_results = self._get_cancelled_lab_result_specs(
                resolve_multiple=resolve_multiple, interval_days=interval_days
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
                + cancelled_lab_results
            )

        interval_days = [10.0, 30.0, 180.0, 365.0]

        visits = self._get_visits_specs(resolve_multiple=[count], interval_days=interval_days)

        admissions = self._get_admissions_specs(
            resolve_multiple=[count, summed], interval_days=interval_days
        )

        diagnoses = self._get_diagnoses_specs(
            resolve_multiple=[count, boolean], interval_days=interval_days
        )

        medications = self._get_medication_specs(
            resolve_multiple=[count, boolean], interval_days=interval_days
        )

        beroligende_medicin = self._get_beroligende_medicin_specs(
            resolve_multiple=[count, boolean], interval_days=interval_days
        )

        coercion = self._get_coercion_specs(
            resolve_multiple=[count, summed, boolean], interval_days=interval_days
        )

        structured_sfi = self._get_structured_sfi_specs(
            resolve_multiple=[mean, maximum, minimum],
            interval_days=interval_days,
        )

        lab_results = self._get_lab_result_specs(
            resolve_multiple=[maximum, minimum, mean, latest], interval_days=interval_days
        )

        cancelled_lab_results = self._get_cancelled_lab_result_specs(
            resolve_multiple=[count, boolean], interval_days=interval_days
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
            + cancelled_lab_results
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

        if self.lookbehind_180d_mean:
            log.warning(
                "--- !!! Using all features, but only a lookbehind of 180 days and mean as aggregation function !!! ---"
            )
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
