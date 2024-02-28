"""Describe flattened dataset."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from timeseriesflattener.v1.feature_specs.single_specs import StaticSpec, TemporalSpec
from timeseriesflattener.v1.flattened_dataset import PredictorSpec

from psycop.common.feature_generation.data_checks.flattened.data_integrity import (
    save_feature_set_integrity_checks_from_dir,
)
from psycop.common.feature_generation.data_checks.flattened.feature_describer import (
    save_feature_descriptive_stats_from_dir,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo

log = logging.getLogger(__name__)


def save_flattened_dataset_description_to_disk(
    project_info: ProjectInfo,
    feature_set_dir: Path,
    feature_specs: list[TemporalSpec | StaticSpec],
    splits: Iterable[str] = ("train", "val", "test"),
    compare_splits: bool = True,
):
    """Describe and check flattened dataset. Runs and saves train data integrity checks, split pair integrity checks, outcome integrity checks and a html table containing descriptive statistics for each feature.

    Args:
        project_info (ProjectInfo): Project info
        feature_specs (list[TemporalSpec | StaticSpec]): Feature specs for dataset description.
        feature_set_dir (Path): Feature set directory
        splits (Iterable[str], optional): Splits to include in the integrity checks. Defaults to ("train", "val", "test").
        compare_splits (bool, optional): Whether to compare splits, e.g. do all categories exist in both train and val. Defaults to True.
    """
    feature_set_descriptive_stats_path = feature_set_dir / "feature_set_descriptive_stats"
    data_integrity_checks_path = feature_set_dir / "data_integrity_checks"

    log.info(
        f"Saving flattened dataset descriptions to disk. Check {feature_set_descriptive_stats_path} and {data_integrity_checks_path} to view data set descriptions and validate that your dataset is not broken in some way."
    )

    save_feature_descriptive_stats_from_dir(
        feature_set_dir=feature_set_dir,
        feature_specs=[s for s in feature_specs if isinstance(s, (StaticSpec, PredictorSpec))],
        file_suffix=".parquet",
        prefixes_to_describe=set(project_info.prefix.__dict__.values()),
    )

    save_feature_set_integrity_checks_from_dir(
        feature_set_dir=feature_set_dir,
        splits=splits,
        out_dir=data_integrity_checks_path,
        dataset_format="parquet",
        compare_splits=compare_splits,
    )
