"""Flatten the dataset."""
import logging
from typing import Optional

import pandas as pd
import psutil
from timeseriesflattener.feature_cache.cache_to_disk import DiskCache
from timeseriesflattener.feature_spec_objects import _AnySpec
from timeseriesflattener.flattened_dataset import TimeseriesFlattener

from psycop_feature_generation.application_modules.filter_prediction_times import (
    PredictionTimeFilterer,
)
from psycop_feature_generation.application_modules.project_setup import ProjectInfo
from psycop_feature_generation.application_modules.wandb_utils import (
    wandb_alert_on_exception,
)
from psycop_feature_generation.loaders.raw.load_demographic import birthdays

log = logging.getLogger(__name__)


def filter_prediction_times(
    prediction_times_df: pd.DataFrame,
    project_info: ProjectInfo,
    quarantine_df: Optional[pd.DataFrame] = None,
    quarantine_days: Optional[int] = None,
) -> pd.DataFrame:
    """Filter prediction times.

    Args:
        prediction_times_df (pd.DataFrame): Prediction times dataframe.
            Should contain entity_id and timestamp columns with col_names matching those in project_info.col_names.
        project_info (ProjectInfo): Project info.
        quarantine_df (pd.DataFrame, optional): Quarantine dataframe with "timestamp" and "project_info.col_names.id" columns.
        quarantine_days (int, optional): Number of days to quarantine.

    Returns:
        pd.DataFrame: Filtered prediction times dataframe.
    """
    log.info("Filtering prediction times...")

    filterer = PredictionTimeFilterer(
        prediction_times_df=prediction_times_df,
        entity_id_col_name=project_info.col_names.id,
        quarantine_timestamps_df=quarantine_df,
        quarantine_interval_days=quarantine_days,
    )

    return filterer.filter()


@wandb_alert_on_exception
def create_flattened_dataset(
    feature_specs: list[_AnySpec],
    prediction_times_df: pd.DataFrame,
    drop_pred_times_with_insufficient_look_distance: bool,
    project_info: ProjectInfo,
    quarantine_df: Optional[pd.DataFrame] = None,
    quarantine_days: Optional[int] = None,
) -> pd.DataFrame:
    """Create flattened dataset.

    Args:
        feature_specs (list[_AnySpec]): List of feature specifications of any type.
        project_info (ProjectInfo): Project info.
        prediction_times_df (pd.DataFrame): Prediction times dataframe.
            Should contain entity_id and timestamp columns with col_names matching those in project_info.col_names.
        drop_pred_times_with_insufficient_look_distance (bool): Whether to drop prediction times with insufficient look distance.
            See timeseriesflattener tutorial for more info.
        quarantine_df (pd.DataFrame, optional): Quarantine dataframe with "timestamp" and "project_info.col_names.id" columns.
        quarantine_days (int, optional): Number of days to quarantine. Any prediction time within quarantine_days after the timestamps in quarantine_df will be dropped.

    Returns:
        FlattenedDataset: Flattened dataset.
    """
    filtered_prediction_times_df = filter_prediction_times(
        prediction_times_df=prediction_times_df,
        project_info=project_info,
        quarantine_df=quarantine_df,
        quarantine_days=quarantine_days,
    )

    flattened_dataset = TimeseriesFlattener(
        prediction_times_df=filtered_prediction_times_df,
        n_workers=min(
            len(feature_specs),
            psutil.cpu_count(logical=True),
        ),
        cache=DiskCache(
            feature_cache_dir=project_info.project_path / "feature_cache",
        ),
        drop_pred_times_with_insufficient_look_distance=drop_pred_times_with_insufficient_look_distance,
        predictor_col_name_prefix=project_info.prefix.predictor,
        outcome_col_name_prefix=project_info.prefix.outcome,
        timestamp_col_name=project_info.col_names.timestamp,
        entity_id_col_name=project_info.col_names.id,
    )

    flattened_dataset.add_age(
        date_of_birth_df=birthdays(),
        date_of_birth_col_name="date_of_birth",
    )

    flattened_dataset.add_spec(spec=feature_specs)

    return flattened_dataset.get_df()
