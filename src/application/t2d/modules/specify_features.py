import numpy as np

from psycop_feature_generation.timeseriesflattener.feature_spec_objects import (
    AnySpec,
    BaseModel,
    OutcomeGroupSpec,
    OutcomeSpec,
    PredictorGroupSpec,
    PredictorSpec,
    StaticSpec,
)


class SpecSet(BaseModel):
    """A set of unresolved specs, ready for resolving."""

    temporal_predictors: list[PredictorSpec]
    static_predictors: list[StaticSpec]
    outcomes: list[OutcomeSpec]
    metadata: list[AnySpec]


def get_static_predictor_specs():
    """Get static predictor specs."""
    return [
        StaticSpec(
            values_loader="sex_female",
            input_col_name_override="sex_female",
            prefix="pred",
        ),
    ]


def get_metadata_specs():
    """Get metadata specs."""
    return [
        StaticSpec(
            values_loader="t2d",
            input_col_name_override="timestamp",
            output_col_name_override="timestamp_first_t2d_hba1c",
        ),
        StaticSpec(
            values_loader="timestamp_exclusion",
            input_col_name_override="timestamp",
            output_col_name_override="timestamp_exclusion",
        ),
        PredictorSpec(
            values_loader="hba1c",
            fallback=np.nan,
            interval_days=9999,
            resolve_multiple_fn="count",
            allowed_nan_value_prop=0.0,
            prefix="eval",
        ),
    ]


def get_outcome_specs():
    """Get outcome specs."""
    return OutcomeGroupSpec(
        values_loader=["t2d"],
        interval_days=[year * 365 for year in (1, 2, 3, 4, 5)],
        resolve_multiple_fn=["max"],
        fallback=[0],
        incident=[True],
        allowed_nan_value_prop=[0],
    ).create_combinations()


def get_temporal_predictor_specs() -> list[PredictorSpec]:
    """Generate predictor spec list."""
    resolve_multiple = ["max", "min", "mean", "latest", "count"]
    interval_days = [30, 90, 180, 365, 730]
    allowed_nan_value_prop = [0]

    unresolved_temporal_predictor_specs: list[PredictorSpec] = []

    unresolved_temporal_predictor_specs += PredictorGroupSpec(
        values_loader=(
            "hba1c",
            "alat",
            "hdl",
            "ldl",
            "scheduled_glc",
            "unscheduled_p_glc",
            "triglycerides",
            "fasting_ldl",
            "crp",
            "egfr",
            "albumine_creatinine_ratio",
        ),
        resolve_multiple_fn=resolve_multiple,
        interval_days=interval_days,
        fallback=[np.nan],
        allowed_nan_value_prop=allowed_nan_value_prop,
    ).create_combinations()

    unresolved_temporal_predictor_specs += PredictorGroupSpec(
        values_loader=(
            "essential_hypertension",
            "hyperlipidemia",
            "polycystic_ovarian_syndrome",
            "sleep_apnea",
        ),
        resolve_multiple_fn=resolve_multiple,
        interval_days=interval_days,
        fallback=[0],
        allowed_nan_value_prop=allowed_nan_value_prop,
    ).create_combinations()

    unresolved_temporal_predictor_specs += PredictorGroupSpec(
        values_loader=("antipsychotics",),
        interval_days=interval_days,
        resolve_multiple_fn=resolve_multiple,
        fallback=[0],
        allowed_nan_value_prop=allowed_nan_value_prop,
    ).create_combinations()

    unresolved_temporal_predictor_specs += PredictorGroupSpec(
        values_loader=["weight_in_kg", "height_in_cm", "bmi"],
        interval_days=interval_days,
        resolve_multiple_fn=["latest"],
        fallback=[np.nan],
        allowed_nan_value_prop=allowed_nan_value_prop,
    ).create_combinations()

    return unresolved_temporal_predictor_specs


def get_spec_set() -> SpecSet:
    """Get a spec set."""
    return SpecSet(
        temporal_predictors=get_temporal_predictor_specs(),
        static_predictors=get_static_predictor_specs(),
        outcomes=get_outcome_specs(),
        metadata=get_metadata_specs(),
    )
