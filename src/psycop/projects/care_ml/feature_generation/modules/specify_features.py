"""Feature specification module."""
import logging
from typing import Callable, Sequence, Union

import numpy as np
from psycop.common.feature_generation.application_modules.project_setup import (
    ProjectInfo,
)
from psycop.common.feature_generation.loaders.raw.load_coercion import (
    af_helbredsmaessige_grunde,
    af_legemlig_lidelse,
    baelte,
    beroligende_medicin,
    ect,
    farlighed,
    fastholden,
    medicinering,
    paa_grund_af_farlighed,
    remme,
    skema_1,
    skema_2_without_nutrition,
    skema_3,
    tvangsindlaeggelse,
    tvangstilbageholdelse,
)
from psycop.common.feature_generation.loaders.raw.load_demographic import sex_female
from psycop.common.feature_generation.loaders.raw.load_diagnoses import (
    cluster_b,
    essential_hypertension,
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
    gerd,
    hyperlipidemia,
    manic_and_bipolar,
    polycystic_ovarian_syndrome,
    schizoaffective,
    schizophrenia,
    sleep_apnea,
)
from psycop.common.feature_generation.loaders.raw.load_lab_results import (
    alat,
    albumine_creatinine_ratio,
    arterial_p_glc,
    cancelled_standard_lab_results,
    crp,
    egfr,
    fasting_ldl,
    fasting_p_glc,
    hba1c,
    hdl,
    ldl,
    ogtt,
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
    scheduled_glc,
    triglycerides,
    unscheduled_p_glc,
    urinary_glc,
)
from psycop.common.feature_generation.loaders.raw.load_medications import (
    alcohol_abstinence,
    analgesic,
    antidepressives,
    antihypertensives,
    antipsychotics,
    anxiolytics,
    aripiprazole_depot,
    benzodiazepine_related_sleeping_agents,
    benzodiazepines,
    clozapine,
    diuretics,
    first_gen_antipsychotics,
    gerd_drugs,
    haloperidol_depot,
    hyperactive_disorders_medications,
    hypnotics,
    lamotrigine,
    lithium,
    olanzapine,
    olanzapine_depot,
    opioid_dependence,
    paliperidone_depot,
    perphenazine_depot,
    pregabaline,
    risperidone_depot,
    second_gen_antipsychotics,
    selected_nassa,
    snri,
    ssri,
    statins,
    tca,
    top_10_weight_gaining_antipsychotics,
    valproate,
    zuclopenthixol_depot,
)
from psycop.common.feature_generation.loaders.raw.load_structured_sfi import (
    bmi,
    broeset_violence_checklist,
    hamilton_d17,
    height_in_cm,
    mas_m,
    selvmordsrisiko,
    weight_in_kg,
)
from psycop.common.feature_generation.loaders.raw.load_text import load_aktuel_psykisk
from psycop.common.feature_generation.loaders.raw.load_visits import (
    physical_visits,
    physical_visits_to_psychiatry,
    physical_visits_to_somatic,
)
from psycop.common.feature_generation.text_models.utils import load_text_model
from psycop.projects.t2d.feature_generation.outcome_specification.combined import (
    get_first_diabetes_indicator,
)
from psycop.projects.t2d.feature_generation.outcome_specification.lab_results import (
    get_first_diabetes_lab_result_above_threshold,
)
from timeseriesflattener.aggregation_fns import (
    boolean,
    change_per_day,
    concatenate,
    count,
    latest,
    maximum,
    mean,
    mean_number_of_characters,
    minimum,
    type_token_ratio,
    variance,
)
from timeseriesflattener.feature_specs.group_specs import (
    NamedDataframe,
    OutcomeGroupSpec,
    PredictorGroupSpec,
    TextPredictorGroupSpec,
)
from timeseriesflattener.feature_specs.single_specs import (
    AnySpec,
    BaseModel,
    OutcomeSpec,
    PredictorSpec,
    StaticSpec,
    TextPredictorSpec,
)
from timeseriesflattener.text_embedding_functions import (
    sklearn_embedding,
)

log = logging.getLogger(__name__)


class SpecSet(BaseModel):
    """A set of unresolved specs, ready for resolving."""

    temporal_predictors: list[PredictorSpec]
    static_predictors: list[StaticSpec]


class FeatureSpecifier:
    """Specify features based on prediction time."""

    def __init__(self, project_info: ProjectInfo, min_set_for_debug: bool = False):
        self.min_set_for_debug = min_set_for_debug
        self.project_info = project_info

    def _get_static_predictor_specs(self) -> list[StaticSpec]:
        """Get static predictor specs."""
        return [
            StaticSpec(
                timeseries_df=sex_female(),
                feature_base_name="sex_female",
                prefix=self.project_info.prefix.predictor,
            ),
        ]

    def _get_weight_and_height_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get weight and height specs."""
        log.info("-------- Generating weight and height specs --------")

        weight_height_bmi = PredictorGroupSpec(
            named_dataframes=[
                NamedDataframe(df=weight_in_kg(), name="weight_in_kg"),
                NamedDataframe(df=height_in_cm(), name="height_in_cm"),
                NamedDataframe(df=bmi(), name="bmi"),
            ],
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[np.nan],
            prefix=self.project_info.prefix.predictor,
        ).create_combinations()

        return weight_height_bmi

    def _get_visits_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get visits specs."""
        log.info("-------- Generating visits specs --------")

        visits = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=physical_visits(), name="physical_visits"),
                NamedDataframe(
                    df=physical_visits_to_psychiatry(),
                    name="physical_visits_to_psychiatry",
                ),
                NamedDataframe(
                    df=physical_visits_to_somatic(), name="physical_visits_to_somatic"
                ),
            ),
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[0],
        ).create_combinations()

        return visits

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
                NamedDataframe(df=f3_disorders(), name="f3_disorders"),
                NamedDataframe(df=f4_disorders(), name="f4_disorders"),
                NamedDataframe(df=f5_disorders(), name="f5_disorders"),
                NamedDataframe(df=f6_disorders(), name="f6_disorders"),
                NamedDataframe(df=f7_disorders(), name="f7_disorders"),
                NamedDataframe(df=f8_disorders(), name="f8_disorders"),
                NamedDataframe(df=f9_disorders(), name="f9_disorders"),
            ),
            aggregation_fns=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[0],
        ).create_combinations()

        return psychiatric_diagnoses

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
                NamedDataframe(df=anxiolytics(), name="anxiolytics"),
                NamedDataframe(df=hypnotics(), name="hypnotics and sedatives"),
                NamedDataframe(df=antidepressives(), name="antidepressives"),
                NamedDataframe(df=lithium(), name="lithium"),
                NamedDataframe(df=alcohol_abstinence(), name="alcohol_abstinence"),
                NamedDataframe(df=opioid_dependence(), name="opioid_dependency"),
            ),
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[0],
        ).create_combinations()

        return psychiatric_medications

    def _get_schema_1_and_2_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get schema 1 and schema 2 coercion specs."""
        log.info("-------- Generating schema 1 and schema 2 coercion specs --------")

        coercion = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=skema_1(), name="skema_1"),
                NamedDataframe(df=tvangsindlaeggelse(), name="tvangsindlaeggelse"),
                NamedDataframe(
                    df=tvangstilbageholdelse(), name="tvangstilbageholdelse"
                ),
                NamedDataframe(
                    df=paa_grund_af_farlighed(), name="paa_grund_af_farlighed"
                ),  # røde papirer
                NamedDataframe(
                    df=af_helbredsmaessige_grunde(), name="af_helbredsmaessige_grunde"
                ),  # gule papirer
                NamedDataframe(
                    df=skema_2_without_nutrition(), name="skema_2_without_nutrition"
                ),
                NamedDataframe(df=medicinering(), name="medicinering"),
                NamedDataframe(df=ect(), name="ect"),
                NamedDataframe(df=af_legemlig_lidelse(), name="af_legemlig_lidelse"),
            ),
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[0],
        ).create_combinations()

        return coercion

    def _get_schema_1_and_2_current_status_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get "current" status of schema 1 and schema 2 coercion."""
        log.info(
            "Generating schema 1 and schema 2 current status specs --------",
        )

        coercion = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=skema_1(unpack_to_intervals=True), name="skema_1"),
                NamedDataframe(
                    df=tvangsindlaeggelse(unpack_to_intervals=True),
                    name="tvangsindlaeggelse",
                ),
                NamedDataframe(
                    df=tvangstilbageholdelse(unpack_to_intervals=True),
                    name="tvangstilbageholdelse",
                ),
                NamedDataframe(
                    df=paa_grund_af_farlighed(unpack_to_intervals=True),
                    name="paa_grund_af_farlighed",
                ),  # røde papirer
                NamedDataframe(
                    df=af_helbredsmaessige_grunde(unpack_to_intervals=True),
                    name="af_helbredsmaessige_grunde",
                ),  # gule papirer
                NamedDataframe(
                    df=skema_2_without_nutrition(unpack_to_intervals=True),
                    name="skema_2_without_nutrition",
                ),
                NamedDataframe(
                    df=medicinering(unpack_to_intervals=True), name="medicinering"
                ),
                NamedDataframe(df=ect(unpack_to_intervals=True), name="ect"),
                NamedDataframe(
                    df=af_legemlig_lidelse(unpack_to_intervals=True),
                    name="af_legemlig_lidelse",
                ),
            ),
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[0],
        ).create_combinations()

        return coercion

    def _get_schema_3_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get schema 3 coercion specs."""
        log.info("-------- Generating schema 3 coercion specs --------")

        coercion = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=skema_3(), name="skema_3"),
                NamedDataframe(df=fastholden(), name="fastholden"),
                NamedDataframe(df=baelte(), name="baelte"),
                NamedDataframe(df=remme(), name="remme"),
                NamedDataframe(df=farlighed(), name="farlighed"),
            ),
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[0],
        ).create_combinations()

        return coercion

    def _get_forced_medication_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get forced medication coercion specs."""
        log.info("-------- Generating forced medication coercion specs --------")

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
                    df=broeset_violence_checklist(), name="broeset_violence_checklist"
                ),
                NamedDataframe(df=selvmordsrisiko(), name="selvmordsrisiko"),
                NamedDataframe(df=hamilton_d17(), name="hamilton_d17"),
                NamedDataframe(df=mas_m(), name="mas_m"),
            ),
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[0],
        ).create_combinations()

        return structured_sfi

    def _get_text_features_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get mean character length sfis specs"""
        log.info("-------- Generating mean character length all sfis specs --------")

        text_features = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=load_aktuel_psykisk(), name="aktuelt_psykisk"),
            ),
            lookbehind_days=interval_days,
            fallback=[np.nan],
            aggregation_fns=resolve_multiple,
        ).create_combinations()

        return text_features

    def _get_text_embedding_features_specs(
        self,
        resolve_multiple: Callable,
        interval_days: Sequence[float],
    ) -> list[TextPredictorSpec]:
        """Get bow all sfis specs"""
        log.info("-------- Generating bow all sfis specs --------")

        bow_model = load_text_model(
            filename="bow_psycop_train_all_sfis_preprocessed_sfi_type_Aktueltpsykisk_ngram_range_12_max_df_10_min_df_1_max_features_500.pkl",
        )
        tfidf_model = load_text_model(
            filename="tfidf_psycop_train_all_sfis_preprocessed_sfi_type_Aktueltpsykisk_ngram_range_12_max_df_10_min_df_1_max_features_500.pkl",
        )

        return_list = []

        for i in interval_days:
            return_list.append(
                [
                    TextPredictorSpec(
                        timeseries_df=load_aktuel_psykisk(),
                        lookbehind_days=i,
                        feature_base_name="bow",
                        embedding_fn=[sklearn_embedding],  # type: ignore
                        embedding_fn_kwargs=[{"model": bow_model}],  # type: ignore
                        fallback=np.nan,
                    ),
                    TextPredictorSpec(
                        timeseries_df=load_aktuel_psykisk(),
                        lookbehind_days=i,
                        feature_base_name="tfidf_model",
                        embedding_fn=[sklearn_embedding],  # type: ignore
                        embedding_fn_kwargs=[{"model": tfidf_model}],  # type: ignore
                        fallback=np.nan,
                    ),
                ]
            )

        return return_list

    def _get_temporal_predictor_specs(self) -> list[PredictorSpec]:
        """Generate predictor spec list."""
        log.info("-------- Generating temporal predictor specs --------")

        if self.min_set_for_debug:
            return [
                PredictorSpec(
                    feature_base_name="f0_disorders",
                    timeseries_df=f0_disorders(),
                    lookbehind_days=100,
                    aggregation_fn=boolean,
                    fallback=np.nan,
                    prefix=self.project_info.prefix.predictor,
                ),
            ]

        resolve_multiple = [boolean, count]
        interval_days = [10.0, 30.0, 180.0, 365.0, 730.0]

        latest_weight_height_bmi = self._get_weight_and_height_specs(
            resolve_multiple=[latest],
            interval_days=interval_days,
        )

        visits = self._get_visits_specs(
            resolve_multiple=[*resolve_multiple, sum],
            interval_days=interval_days,
        )

        diagnoses = self._get_diagnoses_specs(
            resolve_multiple=[boolean],
            interval_days=interval_days,
        )

        medications = self._get_medication_specs(
            resolve_multiple=resolve_multiple,
            interval_days=[1, 3, 7, *interval_days],
        )

        schema_1_schema_2_coercion = self._get_schema_1_and_2_specs(
            resolve_multiple=[*resolve_multiple, sum],
            interval_days=[7, *interval_days],
        )

        schema_1_schema_2_coercion_current_status = (
            self._get_schema_1_and_2_current_status_specs(
                resolve_multiple=[boolean],
                interval_days=[1, 3],
            )
        )

        schema_3_coercion = self._get_schema_3_specs(
            resolve_multiple=[*resolve_multiple, sum],
            interval_days=[730],
        )

        forced_medication_coercion = self._get_forced_medication_specs(
            resolve_multiple=resolve_multiple,
            interval_days=[730],
        )

        structured_sfi = self._get_structured_sfi_specs(
            resolve_multiple=[mean, max, min, change_per_day, variance],
            interval_days=[1, 3, 7, *interval_days],
        )

        return (
            latest_weight_height_bmi
            + visits
            + medications
            + diagnoses
            + schema_1_schema_2_coercion
            + schema_1_schema_2_coercion_current_status
            + schema_3_coercion
            + forced_medication_coercion
            + structured_sfi
        )

    def _get_text_predictor_specs(
        self,
    ) -> list[Union[TextPredictorSpec, PredictorSpec]]:
        """Generate text predictor spec list."""
        log.info("-------- Generating text predictor specs --------")

        if self.min_set_for_debug:
            return self._get_text_features_specs(
                resolve_multiple=[mean_number_of_characters],
                interval_days=[7],
            )  # type: ignore

        resolve_multiple = [concatenate]
        interval_days = [7.0, 30.0]

        text_features = self._get_text_features_specs(
            resolve_multiple=[mean_number_of_characters, type_token_ratio],
            interval_days=interval_days,
        )

        text_embedding_features = self._get_text_embedding_features_specs(
            resolve_multiple=concatenate,
            interval_days=interval_days,
        )

        return text_features + text_embedding_features

    def get_feature_specs(
        self,
    ) -> list[Union[TextPredictorSpec, StaticSpec, PredictorSpec]]:
        """Get a spec set."""

        if self.min_set_for_debug:
            return self._get_text_predictor_specs()

        return (
            self._get_static_predictor_specs()
            + self._get_temporal_predictor_specs()
            + self._get_text_predictor_specs()
        )
