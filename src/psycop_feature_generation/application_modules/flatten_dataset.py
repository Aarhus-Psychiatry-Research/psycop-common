"""Flatten the dataset."""
import pandas as pd
import psutil
from timeseriesflattener.feature_cache.cache_to_disk import DiskCache
from timeseriesflattener.feature_spec_objects import AnySpec
from timeseriesflattener.flattened_dataset import TimeseriesFlattener

from psycop_feature_generation.application_modules.project_setup import ProjectInfo
from psycop_feature_generation.application_modules.wandb_utils import (
    wandb_alert_on_exception,
)
from psycop_feature_generation.loaders.raw.load_demographic import birthdays


@wandb_alert_on_exception
def create_flattened_dataset(
    feature_specs: list[AnySpec],
    prediction_times_df: pd.DataFrame,
    drop_pred_times_with_insufficient_look_distance: bool,
    project_info: ProjectInfo,
) -> pd.DataFrame:
    """Create flattened dataset.

    Args:
        feature_specs (list[AnySpec]): List of feature specifications of any type.
        project_info (ProjectInfo): Project info.
        prediction_times_df (pd.DataFrame): Prediction times dataframe.
            Should contain entity_id and timestamp columns with col_names matching those in project_info.col_names.
        drop_pred_times_with_insufficient_look_distance (bool): Whether to drop prediction times with insufficient look distance.
            See timeseriesflattener tutorial for more info.

    Returns:
        FlattenedDataset: Flattened dataset.
    """

    flattened_dataset = TimeseriesFlattener(
        prediction_times_df=prediction_times_df,
        n_workers=min(
            len(feature_specs),
            psutil.cpu_count(logical=False),
        ),
        cache=DiskCache(
            feature_cache_dir=project_info.feature_set_path / "feature_cache",
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

    return flattened_dataset.get_df()
