"""Feature specification module."""
import logging
from typing import Union

import numpy as np
from psycop.common.feature_generation.application_modules.project_setup import (
    ProjectInfo,
)
from psycop.common.feature_generation.text_models.utils import load_text_model
from timeseriesflattener.feature_spec_objects import (
    BaseModel,
    PredictorGroupSpec,
    PredictorSpec,
    StaticSpec,
    TextPredictorGroupSpec,
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
                values_loader="sex_female",  # type: ignore
                input_col_name_override="sex_female",
                prefix=self.project_info.prefix.predictor,
                feature_name="sex_female",
            ),
        ]

    def _get_weight_and_height_specs(
        self,
        resolve_multiple: list[str],
        interval_days: list[float],
        allowed_nan_value_prop: list[float],
    ) -> list[PredictorSpec]:
        """Get weight and height specs."""
        log.info("-------- Generating weight and height specs --------")

        weight_height_bmi = PredictorGroupSpec(
            values_loader=["weight_in_kg", "height_in_cm", "bmi"],
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,  # type: ignore
            fallback=[np.nan],
            allowed_nan_value_prop=allowed_nan_value_prop,
            prefix=self.project_info.prefix.predictor,
        ).create_combinations()

        return weight_height_bmi

    def _get_visits_specs(
        self,
        resolve_multiple: list[str],
        interval_days: list[float],
        allowed_nan_value_prop: list[float],
    ) -> list[PredictorSpec]:
        """Get visits specs."""
        log.info("-------- Generating visits specs --------")

        visits = PredictorGroupSpec(
            values_loader=(
                "physical_visits",
                "physical_visits_to_psychiatry",
                "physical_visits_to_somatic",
            ),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,  # type: ignore
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return visits

    def _get_diagnoses_specs(
        self,
        resolve_multiple: list[str],
        interval_days: list[float],
        allowed_nan_value_prop: list[float],
    ) -> list[PredictorSpec]:
        """Get diagnoses specs."""
        log.info("-------- Generating diagnoses specs --------")

        psychiatric_diagnoses = PredictorGroupSpec(
            values_loader=(
                "f0_disorders",
                "f1_disorders",
                "f2_disorders",
                "f3_disorders",
                "f4_disorders",
                "f5_disorders",
                "f6_disorders",
                "f7_disorders",
                "f8_disorders",
                "f9_disorders",
            ),
            resolve_multiple_fn=resolve_multiple,  # type: ignore
            lookbehind_days=interval_days,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return psychiatric_diagnoses

    def _get_medication_specs(
        self,
        resolve_multiple: list[str],
        interval_days: list[float],
        allowed_nan_value_prop: list[float],
    ) -> list[PredictorSpec]:
        """Get medication specs."""
        log.info("-------- Generating medication specs --------")

        psychiatric_medications = PredictorGroupSpec(
            values_loader=(
                "antipsychotics",
                "anxiolytics",
                "hypnotics and sedatives",
                "antidepressives",
                "lithium",
                "alcohol_abstinence",
                "opioid_dependency",
            ),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,  # type: ignore
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return psychiatric_medications

    def _get_schema_1_and_2_specs(
        self,
        resolve_multiple: list[str],
        interval_days: list[float],
        allowed_nan_value_prop: list[float],
    ) -> list[PredictorSpec]:
        """Get schema 1 and schema 2 coercion specs."""
        log.info("-------- Generating schema 1 and schema 2 coercion specs --------")

        coercion = PredictorGroupSpec(
            values_loader=(
                "skema_1",
                "tvangsindlaeggelse",
                "tvangstilbageholdelse",
                "paa_grund_af_farlighed",  # røde papirer
                "af_helbredsmaessige_grunde",  # gule papirer
                "skema_2_without_nutrition",
                "medicinering",
                "ect",
                "af_legemlig_lidelse",
            ),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,  # type: ignore
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return coercion

    def _get_schema_1_and_2_current_status_specs(
        self,
        resolve_multiple: list[str],
        interval_days: list[float],
        allowed_nan_value_prop: list[float],
    ) -> list[PredictorSpec]:
        """Get "current" status of schema 1 and schema 2 coercion."""
        log.info(
            "-------- Generating schema 1 and schema 2 current status specs --------",
        )

        coercion = PredictorGroupSpec(
            values_loader=(
                "skema_1",
                "tvangsindlaeggelse",
                "tvangstilbageholdelse",
                "paa_grund_af_farlighed",  # røde papirer
                "af_helbredsmaessige_grunde",  # gule papirer
                "skema_2_without_nutrition",
                "medicinering",
                "ect",
                "af_legemlig_lidelse",
            ),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,  # type: ignore
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
            loader_kwargs=[{"unpack_to_intervals": True}],
        ).create_combinations()

        return coercion

    def _get_schema_3_specs(
        self,
        resolve_multiple: list[str],
        interval_days: list[float],
        allowed_nan_value_prop: list[float],
    ) -> list[PredictorSpec]:
        """Get schema 3 coercion specs."""
        log.info("-------- Generating schema 3 coercion specs --------")

        coercion = PredictorGroupSpec(
            values_loader=(
                "skema_3",
                "fastholden",
                "baelte",
                "remme",
                "farlighed",
            ),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,  # type: ignore
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return coercion

    def _get_forced_medication_specs(
        self,
        resolve_multiple: list[str],
        interval_days: list[float],
        allowed_nan_value_prop: list[float],
    ) -> list[PredictorSpec]:
        """Get forced medication coercion specs."""
        log.info("-------- Generating forced medication coercion specs --------")

        beroligende_medicin = PredictorGroupSpec(
            values_loader=("beroligende_medicin",),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,  # type: ignore
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return beroligende_medicin

    def _get_structured_sfi_specs(
        self,
        resolve_multiple: list[str],
        interval_days: list[float],
        allowed_nan_value_prop: list[float],
    ) -> list[PredictorSpec]:
        """Get structured sfi specs."""
        log.info("-------- Generating structured sfi specs --------")

        structured_sfi = PredictorGroupSpec(
            values_loader=(
                "broeset_violence_checklist",
                "selvmordsrisiko",
                "hamilton_d17",
                "mas_m",
            ),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,  # type: ignore
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return structured_sfi

    def _get_text_features_specs(
        self,
        resolve_multiple: list[str],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get mean character length sfis specs"""
        log.info("-------- Generating mean character length all sfis specs --------")

        text_features = PredictorGroupSpec(
            values_loader=("aktuelt_psykisk",),
            lookbehind_days=interval_days,
            fallback=[np.nan],
            resolve_multiple_fn=resolve_multiple,  # type: ignore
        ).create_combinations()

        return text_features

    def _get_text_embedding_features_specs(
        self,
        resolve_multiple: list[str],
        interval_days: list[float],
    ) -> list[TextPredictorSpec]:
        """Get bow all sfis specs"""
        log.info("-------- Generating bow all sfis specs --------")

        bow_model = load_text_model(
            filename="bow_psycop_train_all_sfis_preprocessed_sfi_type_Aktueltpsykisk_ngram_range_12_max_df_10_min_df_1_max_features_500.pkl",
        )
        tfidf_model = load_text_model(
            filename="tfidf_psycop_train_all_sfis_preprocessed_sfi_type_Aktueltpsykisk_ngram_range_12_max_df_10_min_df_1_max_features_500.pkl",
        )

        return TextPredictorGroupSpec(
            values_loader=("aktuelt_psykisk",),
            lookbehind_days=interval_days,
            fallback=[np.nan],
            resolve_multiple_fn=resolve_multiple,  # type: ignore
            embedding_fn=[sklearn_embedding],
            embedding_fn_kwargs=[{"model": bow_model}, {"model": tfidf_model}],
        ).create_combinations()

    def _get_temporal_predictor_specs(self) -> list[PredictorSpec]:
        """Generate predictor spec list."""
        log.info("-------- Generating temporal predictor specs --------")

        if self.min_set_for_debug:
            return [
                PredictorSpec(
                    values_loader="f0_disorders",
                    lookbehind_days=100,
                    resolve_multiple_fn="bool",
                    fallback=np.nan,
                    allowed_nan_value_prop=0,
                    prefix=self.project_info.prefix.predictor,
                ),
            ]

        resolve_multiple = ["bool", "count"]
        interval_days = [10.0, 30.0, 180.0, 365.0, 730.0]
        allowed_nan_value_prop = [0.0]

        latest_weight_height_bmi = self._get_weight_and_height_specs(
            resolve_multiple=["latest"],
            interval_days=interval_days,
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        visits = self._get_visits_specs(
            resolve_multiple=[*resolve_multiple, "sum"],
            interval_days=interval_days,
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        diagnoses = self._get_diagnoses_specs(
            resolve_multiple=["bool"],
            interval_days=interval_days,
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        medications = self._get_medication_specs(
            resolve_multiple=resolve_multiple,
            interval_days=[1, 3, 7, *interval_days],
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        schema_1_schema_2_coercion = self._get_schema_1_and_2_specs(
            resolve_multiple=[*resolve_multiple, "sum"],
            interval_days=[7, *interval_days],
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        schema_1_schema_2_coercion_current_status = (
            self._get_schema_1_and_2_current_status_specs(
                resolve_multiple=["bool"],
                interval_days=[1, 3],
                allowed_nan_value_prop=allowed_nan_value_prop,
            )
        )

        schema_3_coercion = self._get_schema_3_specs(
            resolve_multiple=[*resolve_multiple, "sum"],
            interval_days=[730],
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        forced_medication_coercion = self._get_forced_medication_specs(
            resolve_multiple=resolve_multiple,
            interval_days=[730],
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        structured_sfi = self._get_structured_sfi_specs(
            resolve_multiple=["mean", "max", "min", "change_per_day", "variance"],
            interval_days=[1, 3, 7, *interval_days],
            allowed_nan_value_prop=allowed_nan_value_prop,
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
                resolve_multiple=["mean_len"],
                interval_days=[7],
            )

        resolve_multiple = ["concatenate"]
        interval_days = [7.0, 30.0]

        text_features = self._get_text_features_specs(
            resolve_multiple=["mean_len", "type_token_ratio"],
            interval_days=interval_days,
        )

        text_embedding_features = self._get_text_embedding_features_specs(
            resolve_multiple=resolve_multiple,
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
