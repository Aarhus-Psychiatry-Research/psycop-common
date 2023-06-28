"""Feature specification module."""
import logging
from typing import Callable, Sequence  # noqa

import numpy as np
from timeseriesflattener.aggregation_fns import count, latest, maximum, mean, minimum, concatenate
from timeseriesflattener.feature_specs.group_specs import (
    NamedDataframe,
    OutcomeGroupSpec,
    PredictorGroupSpec,
    TextPredictorGroupSpec
)
from timeseriesflattener.feature_specs.single_specs import (
    AnySpec,
    OutcomeSpec,
    PredictorSpec,
    StaticSpec,
    TextPredictorSpec
)
import polars as pl
from psycop.common.feature_generation.application_modules.project_setup import (
    ProjectInfo,
)
from timeseriesflattener.text_embedding_functions import (
    sklearn_embedding,
    sentence_transformers_embedding
)
from psycop.common.feature_generation.loaders.raw.load_demographic import sex_female
from psycop.common.feature_generation.loaders.raw.load_diagnoses import (
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
    polycystic_ovarian_syndrome,
    sleep_apnea,
)
from psycop.common.feature_generation.loaders.raw.load_text import load_aktuel_psykisk
from psycop.common.feature_generation.loaders.raw.load_lab_results import (
    alat,
    albumine_creatinine_ratio,
    arterial_p_glc,
    crp,
    egfr,
    fasting_ldl,
    fasting_p_glc,
    hba1c,
    hdl,
    ldl,
    ogtt,
    scheduled_glc,
    triglycerides,
    unscheduled_p_glc,
    urinary_glc,
)
from psycop.common.feature_generation.loaders.raw.load_medications import (
    antihypertensives,
    antipsychotics,
    benzodiazepine_related_sleeping_agents,
    benzodiazepines,
    clozapine,
    diuretics,
    gerd_drugs,
    lamotrigine,
    lithium,
    pregabaline,
    selected_nassa,
    snri,
    ssri,
    statins,
    tca,
    top_10_weight_gaining_antipsychotics,
    valproate,
)
from psycop.common.feature_generation.loaders.raw.load_structured_sfi import (
    bmi,
    height_in_cm,
    weight_in_kg,
)
from psycop.common.feature_generation.text_models.utils import load_text_model
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.scz_bp_prediction_time_loader import SczBpCohort
from psycop.projects.scz_bp.feature_generation.outcome_specification.first_scz_or_bp_diagnosis import (
    get_first_scz_bp_diagnosis_after_washin,
)
from psycop.projects.t2d.feature_generation.outcome_specification.combined import (
    get_first_diabetes_indicator,
)
from psycop.projects.t2d.feature_generation.outcome_specification.lab_results import (
    get_first_diabetes_lab_result_above_threshold,
)

log = logging.getLogger(__name__)


class FeatureSpecifier:
    """Feature specification class."""

    def __init__(
        self,
        project_info: ProjectInfo,
        min_set_for_debug: bool = False,
    ) -> None:
        self.min_set_for_debug = min_set_for_debug
        self.project_info = project_info

    def _get_static_predictor_specs(self) -> list[StaticSpec]:
        """Get static predictor specs."""
        return [
            StaticSpec(
                feature_base_name="sex_female",
                timeseries_df=sex_female(),
                prefix=self.project_info.prefix.predictor,
            ),
        ]

    def _get_metadata_specs(self) -> list[AnySpec]:
        """Get metadata specs."""
        log.info("-------- Generating metadata specs --------")

        if self.min_set_for_debug:
            return [
                StaticSpec(
                    feature_base_name="first_scz_or_bp_indicator",
                    timeseries_df=get_first_scz_bp_diagnosis_after_washin().to_pandas(),
                    prefix="",
                ),
            ]

        return [
            StaticSpec(
                feature_base_name="first_scz_or_bp_indicator",
                timeseries_df=get_first_scz_bp_diagnosis_after_washin().to_pandas(),
                prefix="",
            ),
            StaticSpec(
                feature_base_name="scz_or_bp_indicator",
                timeseries_df=get_first_scz_bp_diagnosis_after_washin().select(
            pl.col("dw_ek_borger"), pl.col("timestamp"), pl.col("source").alias("value")
                ).to_pandas(),
                prefix="",
            ),
        ]

    def _get_outcome_specs(self) -> list[OutcomeSpec]:
        """Get outcome specs."""
        log.info("-------- Generating outcome specs --------")

        if self.min_set_for_debug:
            return [
                OutcomeSpec(
                    feature_base_name="first_scz_or_bp",
                    timeseries_df=SczBpCohort.get_outcome_timestamps().to_pandas(),
                    lookahead_days=365,
                    aggregation_fn=maximum,
                    fallback=0,
                    incident=True,
                    prefix=self.project_info.prefix.outcome,
                ),
            ]

        return OutcomeGroupSpec(
            named_dataframes=[
                NamedDataframe(
                    df=SczBpCohort.get_outcome_timestamps().to_pandas(),
                    name="first_scz_or_bp",
                ),
            ],
            lookahead_days=[year * 365 for year in (1, 2, 3, 4, 5)],
            aggregation_fns=[maximum],
            fallback=[0],
            incident=[True],
            prefix=self.project_info.prefix.outcome,
        ).create_combinations()

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
                NamedDataframe(df=clozapine(), name="clozapine"),
                NamedDataframe(df=lithium(), name="lithium"),
                NamedDataframe(df=valproate(), name="valproate"),
                NamedDataframe(df=lamotrigine(), name="lamotrigine"),
                NamedDataframe(df=benzodiazepines(), name="benzodiazepines"),
                NamedDataframe(df=pregabaline(), name="pregabaline"),
                NamedDataframe(df=ssri(), name="ssri"),
                NamedDataframe(df=snri(), name="snri"),
                NamedDataframe(df=tca(), name="tca"),
                NamedDataframe(df=selected_nassa(), name="selected_nassa"),
                NamedDataframe(
                    df=benzodiazepine_related_sleeping_agents(),
                    name="benzodiazepine_related_sleeping_agents",
                ),
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

    def _get_text_specs(
            self,
            resolve_multiple: list[Callable],
            interval_days: list[float],
    ) -> list[TextPredictorSpec]:
        log.info("-------- Generating text specs --------")
        
        tfidf_model = load_text_model(
            filename="tfidf_psycop_train_all_sfis_preprocessed_sfi_type_Aktueltpsykisk_ngram_range_12_max_df_10_min_df_1_max_features_500.pkl",
        )

        if self.min_set_for_debug:
            return [
                TextPredictorSpec(
                timeseries_df=load_aktuel_psykisk(),
                lookbehind_days=365,
                feature_base_name="tfidf_aktuel_psykisk",
                embedding_fn=sklearn_embedding,
                embedding_fn_kwargs={"model" : tfidf_model},
                fallback=np.nan,
                )
            ]
        tfidf_specs = TextPredictorGroupSpec(
            named_dataframes=[NamedDataframe(df=load_aktuel_psykisk(), name="aktuel_psykisk")],
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            embedding_fn_name="tfidf",
            fallback=[0],
            embedding_fn=[sklearn_embedding],
            embedding_fn_kwargs=[{"model" : tfidf_model}],
        ).create_combinations()

        sentence_transformer_specs = TextPredictorGroupSpec(
            named_dataframes=[NamedDataframe(df=load_aktuel_psykisk(), name="aktuel_psykisk")],
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            embedding_fn_name="sentence_transformer",
            fallback=[0],
            embedding_fn=[sentence_transformers_embedding],
            embedding_fn_kwargs=[{"model_name" : "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}],
        ).create_combinations()

        return tfidf_specs + sentence_transformer_specs


    def _get_temporal_predictor_specs(self) -> list[PredictorSpec]:
        """Generate predictor spec list."""
        log.info("-------- Generating temporal predictor specs --------")

        if self.min_set_for_debug:
            return [
                PredictorSpec(
                    feature_base_name="hba1c",
                    timeseries_df=hba1c(),
                    lookbehind_days=9999,
                    aggregation_fn=max,
                    fallback=np.nan,
                    prefix=self.project_info.prefix.predictor,
                ),
            ]

        resolve_multiple = [maximum, minimum, mean, latest]
        interval_days: list[float] = [30, 180, 365, 730, 1095, 1460, 1825]

        diagnoses = self._get_diagnoses_specs(
            resolve_multiple,
            interval_days,
        )

        medications = self._get_medication_specs(
            resolve_multiple,
            interval_days,
        )

        text = self._get_text_specs(
            resolve_multiple=[concatenate],
            interval_days=[60, 365, 730],
        )

        return medications + diagnoses + text

    def get_feature_specs(self) -> list[AnySpec]:
        """Get a spec set."""

        if self.min_set_for_debug:
            log.warning(
                "--- !!! Using the minimum set of features for debugging !!! ---",
            )
            return (
                self._get_temporal_predictor_specs()
                + self._get_outcome_specs()
                + self._get_metadata_specs()
            )

        return (
            self._get_temporal_predictor_specs()
            + self._get_static_predictor_specs()
            + self._get_outcome_specs()
            + self._get_metadata_specs()
        )
