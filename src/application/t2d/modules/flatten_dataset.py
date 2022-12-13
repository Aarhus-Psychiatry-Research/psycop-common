import time
from pathlib import Path

import pandas as pd
import psutil
from timeseriesflattener.feature_cache.cache_to_disk import DiskCache
from timeseriesflattener.feature_spec_objects import AnySpec
from timeseriesflattener.flattened_dataset import TimeseriesFlattener

from application.t2d.modules.project_setup import ProjectInfo
from psycop_feature_generation.loaders.raw.load_demographic import birthdays
from psycop_feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)


def create_flattened_dataset(
    feature_specs: list[AnySpec],
    project_info: ProjectInfo,
) -> pd.DataFrame:
    """Create flattened dataset.

    Args:
        feature_specs (list[AnySpec]): List of feature specifications of any type.
        project_info (ProjectInfo): Project info.

    Returns:
        FlattenedDataset: Flattened dataset.
    """

    flattened_dataset = TimeseriesFlattener(
        prediction_times_df=physical_visits_to_psychiatry(),
        n_workers=min(
            len(feature_specs),
            psutil.cpu_count(logical=False),
        ),
        cache=DiskCache(
            feature_cache_dir=project_info.feature_set_path / "feature_cache",
        ),
        drop_pred_times_with_insufficient_look_distance=False,
    )

    flattened_dataset.add_age(
        date_of_birth_df=birthdays(),
        date_of_birth_col_name="date_of_birth",
    )

    return flattened_dataset.get_df()
