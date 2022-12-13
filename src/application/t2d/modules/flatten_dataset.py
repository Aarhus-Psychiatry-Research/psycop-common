import time
from pathlib import Path

import pandas as pd
import psutil

from application.t2d.modules.specify_features import SpecSet
from psycop_feature_generation.timeseriesflattener import FlattenedDataset
from psycop_feature_generation.timeseriesflattener.feature_spec_objects import AnySpec, StaticSpec, TemporalSpec, \
    OutcomeSpec, PredictorSpec


def add_metadata_to_ds(
        specs: list[AnySpec],
        flattened_dataset: FlattenedDataset,
) -> FlattenedDataset:
    """Add metadata.

    Args:
        specs (list[AnySpec]): List of specifications.
        flattened_dataset (FlattenedDataset): Flattened dataset.

    Returns:
        FlattenedDataset: Flattened dataset.
    """

    for spec in specs:
        if isinstance(spec, StaticSpec):
            flattened_dataset.add_static_info(
                static_spec=spec,
            )
        elif isinstance(spec, TemporalSpec):
            flattened_dataset.add_temporal_predictor(output_spec=spec)

    return flattened_dataset


def add_outcomes_to_ds(
        flattened_dataset: FlattenedDataset,
        outcome_specs: list[OutcomeSpec],
) -> FlattenedDataset:
    """Add outcomes.

    Args:
        flattened_dataset (FlattenedDataset): Flattened dataset.
        outcome_specs (list[OutcomeSpec]): List of outcome specifications.

    Returns:
        FlattenedDataset: Flattened dataset.
    """

    msg = Printer(timestamp=True)
    msg.info("Adding outcomes")

    for spec in outcome_specs:
        msg.info(f"Adding outcome with {spec.interval_days} days of lookahead")

    msg.good("Finished adding outcomes")

    return flattened_dataset


def add_predictors_to_ds(
        temporal_predictor_specs: list[PredictorSpec],
        static_predictor_specs: list[AnySpec],
        birthdays: pd.DataFrame,
        flattened_dataset: FlattenedDataset,
):
    """Add predictors.

    Args:
        temporal_predictor_specs (list[PredictorSpec]): List of predictor specs.
        static_predictor_specs (list[StaticSpec]): List of static specs.
        birthdays (pd.DataFrame): Birthdays. Used for inferring age at each prediction time.
        flattened_dataset (FlattenedDataset): Flattened dataset.
    """

    msg = Printer(timestamp=True)

    msg.info("Adding static predictors")

    for static_spec in static_predictor_specs:
        flattened_dataset.add_static_info(
            static_spec=static_spec,
        )

    flattened_dataset.add_age_and_date_of_birth(birthdays)

    start_time = time.time()

    msg.info("Adding temporal predictors")
    flattened_dataset.add_temporal_predictors_from_pred_specs(
        predictor_specs=temporal_predictor_specs,
    )

    end_time = time.time()

    # Finish
    msg.good(
        f"Finished adding {len(temporal_predictor_specs)} predictors, took {round((end_time - start_time) / 60, 1)} minutes",
    )

    return flattened_dataset


def create_flattened_dataset(
        prediction_times: pd.DataFrame,
        birthdays: pd.DataFrame,
        spec_set: SpecSet,
        proj_path: Path,
) -> pd.DataFrame:
    """Create flattened dataset.

    Args:
        prediction_times (pd.DataFrame): Dataframe with prediction times.
        birthdays (pd.DataFrame): Birthdays. Used for inferring age at each prediction time.
        spec_set (SpecSet): Set of specifications.
        proj_path (Path): Path to project directory.

    Returns:
        FlattenedDataset: Flattened dataset.
    """
    msg = Printer(timestamp=True)

    msg.info(f"Generating {len(spec_set.temporal_predictors)} features")

    msg.info("Initialising flattened dataset")

    flattened_dataset = FlattenedDataset(
        prediction_times_df=prediction_times,
        n_workers=min(
            len(spec_set.temporal_predictors),
            psutil.cpu_count(logical=False),
        ),
        feature_cache_dir=proj_path / "feature_cache",
    )

    flattened_dataset = add_metadata_to_ds(
        flattened_dataset=flattened_dataset,
        specs=spec_set.metadata,
    )

    flattened_dataset = add_outcomes_to_ds(
        outcome_specs=spec_set.outcomes,
        flattened_dataset=flattened_dataset,
    )

    flattened_dataset = add_predictors_to_ds(
        temporal_predictor_specs=spec_set.temporal_predictors,
        static_predictor_specs=spec_set.static_predictors,
        flattened_dataset=flattened_dataset,
        birthdays=birthdays,
    )

    return flattened_dataset.df
