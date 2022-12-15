"""Feature specification module."""
import logging
from sys import prefix

import numpy as np

from psycop_feature_generation.application_modules.project_setup import ProjectInfo
from timeseriesflattener.feature_spec_objects import (
    BaseModel,
    OutcomeGroupSpec,
    OutcomeSpec,
    PredictorGroupSpec,
    PredictorSpec,
    StaticSpec,
    _AnySpec,
)

from .loaders.t2d_loaders import (  # noqa pylint: disable=unused-import
    timestamp_exclusion,
)

log = logging.getLogger(__name__)


class SpecSet(BaseModel):
    """A set of unresolved specs, ready for resolving."""

    temporal_predictors: list[PredictorSpec]
    static_predictors: list[StaticSpec]
    outcomes: list[OutcomeSpec]
    metadata: list[_AnySpec]


class FeatureSpecifier:
    def __init__(self, project_info: ProjectInfo, min_set_for_debug: bool = False):
        self.min_set_for_debug = min_set_for_debug
        self.project_info = project_info

    def _get_static_predictor_specs(self):
        """Get static predictor specs."""
        return [
            StaticSpec(
                values_loader="sex_female",
                input_col_name_override="sex_female",
                prefix=self.project_info.prefix.predictor,
            ),
        ]

    def _get_metadata_specs(self) -> list[_AnySpec]:
        """Get metadata specs."""
        log.info("–––––––– Generating metadata specs ––––––––")

        return [
            StaticSpec(
                values_loader="t2d",
                input_col_name_override="timestamp",
                output_col_name_override="timestamp_first_t2d_hba1c",
                prefix="",
            ),
            StaticSpec(
                values_loader="timestamp_exclusion",
                input_col_name_override="timestamp",
                output_col_name_override="timestamp_exclusion",
                prefix="",
            ),
            PredictorSpec(
                values_loader="hba1c",
                fallback=np.nan,
                lookbehind_days=9999,
                resolve_multiple_fn="count",
                allowed_nan_value_prop=0.0,
                prefix=self.project_info.prefix.eval,
            ),
        ]

    def _get_outcome_specs(self):
        """Get outcome specs."""
        log.info("–––––––– Generating outcome specs ––––––––")

        return OutcomeGroupSpec(
            values_loader=["t2d"],
            lookahead_days=[year * 365 for year in (1, 2, 3, 4, 5)],
            resolve_multiple_fn=["max"],
            fallback=[0],
            incident=[True],
            allowed_nan_value_prop=[0],
            prefix=self.project_info.prefix.outcome,
        ).create_combinations()

    def _get_medication_specs(
        self, resolve_multiple, interval_days, allowed_nan_value_prop
    ):
        """Get medication specs."""
        log.info("–––––––– Generating medication specs ––––––––")

        psychiatric_medications = PredictorGroupSpec(
            values_loader=(
                "antipsychotics",
                "clozapine",
                "top_10_weight_gaining_antipsychotics",
                "lithium",
                "valproate",
                "lamotrigine",
                "benzodiazepines",
                "pregabaline",
                "ssri",
                "snri",
                "tca",
                "selected_nassa",
                "benzodiazepine_related_sleeping_agents",
            ),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        lifestyle_medications = PredictorGroupSpec(
            values_loader=(
                "gerd_drugs",
                "statins",
                "antihypertensives",
                "diuretics",
            ),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return psychiatric_medications + lifestyle_medications

    def _get_diagnoses_specs(
        self, resolve_multiple, interval_days, allowed_nan_value_prop
    ):
        """Get diagnoses specs."""
        log.info("–––––––– Generating diagnoses specs ––––––––")

        lifestyle_diagnoses = PredictorGroupSpec(
            values_loader=(
                "essential_hypertension",
                "hyperlipidemia",
                "polycystic_ovarian_syndrome",
                "sleep_apnea",
                "gerd",
            ),
            resolve_multiple_fn=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

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
                "hyperkinetic_disorders",
            ),
            resolve_multiple_fn=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return lifestyle_diagnoses + psychiatric_diagnoses

    def _get_lab_result_specs(
        self, resolve_multiple, interval_days, allowed_nan_value_prop
    ):
        """Get lab result specs."""
        log.info("–––––––– Generating lab result specs ––––––––")

        general_lab_results = PredictorGroupSpec(
            values_loader=(
                "alat",
                "hdl",
                "ldl",
                "triglycerides",
                "fasting_ldl",
                "crp",
            ),
            resolve_multiple_fn=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[np.nan],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        diabetes_lab_results = PredictorGroupSpec(
            values_loader=(
                "hba1c",
                "scheduled_glc",
                "unscheduled_p_glc",
                "egfr",
                "albumine_creatinine_ratio",
            ),
            resolve_multiple_fn=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[np.nan],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return general_lab_results + diabetes_lab_results

    def _get_temporal_predictor_specs(self) -> list[PredictorSpec]:
        """Generate predictor spec list."""
        log.info("–––––––– Generating temporal predictor specs ––––––––")

        if self.min_set_for_debug:
            return [
                PredictorSpec(
                    values_loader="hba1c",
                    lookbehind_days=30,
                    resolve_multiple_fn="max",
                    fallback=np.nan,
                    allowed_nan_value_prop=0,
                    prefix=self.project_info.prefix.predictor,
                )
            ]

        resolve_multiple = ["max", "min", "mean", "latest", "count"]
        interval_days = [30, 90, 180, 365, 730]
        allowed_nan_value_prop = [0]

        lab_results = self._get_lab_result_specs(
            resolve_multiple,
            interval_days,
            allowed_nan_value_prop,
        )

        diagnoses = self._get_diagnoses_specs(
            resolve_multiple,
            interval_days,
            allowed_nan_value_prop,
        )

        medications = self._get_medication_specs(
            resolve_multiple,
            interval_days,
            allowed_nan_value_prop,
        )

        demographics = PredictorGroupSpec(
            values_loader=["weight_in_kg", "height_in_cm", "bmi"],
            lookbehind_days=interval_days,
            resolve_multiple_fn=["latest"],
            fallback=[np.nan],
            allowed_nan_value_prop=allowed_nan_value_prop,
            prefix=self.project_info.prefix.predictor,
        ).create_combinations()

        return lab_results + medications + diagnoses + demographics

    def get_feature_specs(self) -> list[_AnySpec]:
        """Get a spec set."""

        if self.min_set_for_debug:
            return self._get_metadata_specs() + self._get_temporal_predictor_specs()

        return (
            self._get_temporal_predictor_specs()
            + self._get_static_predictor_specs()
            + self._get_outcome_specs()
            + self._get_metadata_specs()
        )
