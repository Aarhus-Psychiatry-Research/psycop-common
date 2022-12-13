"""Describe flattened dataset.""" ""
import logging

from psycop_feature_generation.application_modules.project_setup import ProjectInfo
from psycop_feature_generation.application_modules.wandb_utils import (
    wandb_alert_on_exception,
)
from psycop_feature_generation.data_checks.flattened.data_integrity import (
    save_feature_set_integrity_from_dir,
)

log = logging.getLogger(__name__)


@wandb_alert_on_exception
def save_flattened_dataset_description_to_disk(
    project_info: ProjectInfo,
    describe_splits: bool = True,
    compare_splits: bool = True,
):
    """Describe output.

    Args:
        project_info (ProjectInfo): Project info
        describe_splits (bool, optional): Whether to describe each split. Defaults to True.
        compare_splits (bool, optional): Whether to compare splits, e.g. do all categories exist in both train and val. Defaults to True.
    """
    feature_set_description_path = (
        project_info.feature_set_path / "feature_set_description"
    )
    split_feature_distribution_comparison_path = (
        project_info.feature_set_path / "split_feature_distribution_comparison_path"
    )

    log.info(
        f"Saving flattened dataset description to disk. Check {feature_set_description_path} and {split_feature_distribution_comparison_path} to validate that your dataset is not broken in some way.",
    )

    for method in ("describe", "compare"):
        # Describe the feature set
        if method == "describe" and describe_splits:
            splits = ["train"]

            compare_splits_in_method = False
            out_dir = feature_set_description_path

        # Compare train, val and test, to make sure they have the same categories
        if method == "compare" and compare_splits:
            splits = ["train", "val", "test"]

            # Don't describe specs in comparison, to avoid leakage by describing features in val and test
            compare_splits_in_method = True
            out_dir = split_feature_distribution_comparison_path

        save_feature_set_integrity_from_dir(
            splits=splits,
            out_dir=out_dir,
            dataset_format=project_info.dataset_format,
            describe_splits=describe_splits,
            compare_splits=compare_splits_in_method,
            feature_set_dir=project_info.feature_set_path,
        )
