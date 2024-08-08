"""Flatten the dataset."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import psutil
from timeseriesflattener import Flattener
from timeseriesflattener import PredictionTimeFrame as FlattenerPredictionTimeFrame
from timeseriesflattener.v1.flattened_dataset import TimeseriesFlattener

from psycop.common.feature_generation.application_modules.save_dataset_to_disk import (
    split_and_save_dataset_to_disk,
)
from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Sequence
    from pathlib import Path

    import pandas as pd
    import polars as pl
    from timeseriesflattener.v1.feature_specs.single_specs import AnySpec

    from psycop.common.cohort_definition import PredictionTimeFrame
    from psycop.common.feature_generation.application_modules.generate_feature_set import (
        ValueSpecification,
    )
    from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo

log = logging.getLogger(__name__)


def flatten_dataset_to_disk(
    project_info: ProjectInfo,
    feature_specs: list[AnySpec],
    prediction_times_df: pd.DataFrame,
    feature_set_dir: Path,
    split2ids_df: dict[str, pd.DataFrame] | None = None,
    add_birthdays: bool = True,
    split_names: Sequence[str] = ("train", "val", "test"),
):
    flattened_dataset = create_flattened_dataset_tsflattener_v1(
        project_info=project_info,
        feature_specs=feature_specs,
        prediction_times_df=prediction_times_df,
        add_birthdays=add_birthdays,
    )

    split_and_save_dataset_to_disk(
        flattened_df=flattened_dataset,
        project_info=project_info,
        feature_set_dir=feature_set_dir,
        split_ids=split2ids_df,
        split_names=split_names,
    )


def create_flattened_dataset(
    feature_specs: Sequence[ValueSpecification],
    prediction_times_frame: PredictionTimeFrame,
    n_workers: int | None,
    compute_lazily: bool,
    step_size: dt.timedelta | None = None,
) -> pl.DataFrame:
    flattener = Flattener(
        predictiontime_frame=FlattenerPredictionTimeFrame(
            init_df=prediction_times_frame.frame,
            entity_id_col_name=prediction_times_frame.entity_id_col_name,
            timestamp_col_name=prediction_times_frame.timestamp_col_name,
        ),
        compute_lazily=compute_lazily,
        n_workers=n_workers,
    )
    return flattener.aggregate_timeseries(specs=feature_specs, step_size=step_size).df.collect()


def create_flattened_dataset_tsflattener_v1(
    project_info: ProjectInfo,
    feature_specs: list[AnySpec],
    prediction_times_df: pd.DataFrame,
    add_birthdays: bool = True,
    drop_pred_times_with_insufficient_look_distance: bool = False,
) -> pd.DataFrame:
    """Create flattened dataset.

    Args:
        feature_specs (list[AnySpec]): List of feature specifications of any type.
        project_info (ProjectInfo): Project info.
        prediction_times_df (pd.DataFrame): Prediction times dataframe.
            Should contain entity_id and timestamp columns with col_names matching those in project_info.col_names.
        drop_pred_times_with_insufficient_look_distance (bool): Whether to drop prediction times with insufficient look distance.
            See timeseriesflattener tutorial for more info.
        add_birthdays: Whether to add age at prediction time.

    Returns:
        FlattenedDataset: Flattened dataset.
    """

    flattened_dataset = TimeseriesFlattener(
        prediction_times_df=prediction_times_df,
        n_workers=min(len(feature_specs), psutil.cpu_count(logical=True)),
        cache=None,
        drop_pred_times_with_insufficient_look_distance=drop_pred_times_with_insufficient_look_distance,
        predictor_col_name_prefix=project_info.prefix.predictor,
        outcome_col_name_prefix=project_info.prefix.outcome,
        timestamp_col_name=project_info.col_names.timestamp,
        entity_id_col_name=project_info.col_names.id,
    )

    if add_birthdays:
        flattened_dataset.add_age(
            date_of_birth_df=birthdays(), date_of_birth_col_name="date_of_birth"
        )

    flattened_dataset.add_spec(spec=feature_specs)

    return flattened_dataset.get_df()
