"""Describe flattened dataset.""" ""
from __future__ import annotations

import logging
from collections.abc import Iterable

from timeseriesflattener.feature_spec_objects import StaticSpec, TemporalSpec

from psycop_feature_generation.application_modules.project_setup import ProjectInfo
from psycop_feature_generation.application_modules.wandb_utils import (
    wandb_alert_on_exception,
)
from psycop_feature_generation.data_checks.flattened.data_integrity import (
    save_feature_set_integrity_checks_from_dir,
)
from psycop_feature_generation.data_checks.flattened.feature_describer import (
    save_feature_descriptive_stats_from_dir,
)

log = logging.getLogger(__name__)


@wandb_alert_on_exception
def save_flattened_dataset_description_to_disk(
    project_info: ProjectInfo,
    feature_specs: list[TemporalSpec | StaticSpec],
    splits: Iterable[str] = ("train", "val", "test"),
    compare_splits: bool = True,
):
    """Describe and check flattened dataset. Runs and saves train data integrity checks, split pair integrity checks, outcome integrity checks and a html table containing descriptive statistics for each feature.

    Args:
        project_info (ProjectInfo): Project info
        feature_specs (list[TemporalSpec | StaticSpec]): Feature specs for dataset description.
        splits (Iterable[str], optional): Splits to include in the integrity checks. Defaults to ("train", "val", "test").
        compare_splits (bool, optional): Whether to compare splits, e.g. do all categories exist in both train and val. Defaults to True.
    """
    feature_set_descriptive_stats_path = (
        project_info.feature_set_path / "feature_set_descriptive_stats"
    )
    data_integrity_checks_path = project_info.feature_set_path / "data_integrity_checks"

    log.info(
        f"Saving flattened dataset descriptions to disk. Check {feature_set_descriptive_stats_path} and {data_integrity_checks_path} to view data set descriptions and validate that your dataset is not broken in some way.",
    )

    save_feature_descriptive_stats_from_dir(
        feature_set_dir=project_info.feature_set_path,
        feature_specs=feature_specs,
        file_suffix=".parquet",
    )

    save_feature_set_integrity_checks_from_dir(
        feature_set_dir=project_info.feature_set_path,
        splits=splits,
        out_dir=data_integrity_checks_path,
        dataset_format=project_info.dataset_format,
        compare_splits=compare_splits,
    )
