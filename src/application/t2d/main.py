"""Main example on how to generate features.

Uses T2D-features. WIP, will be migrated to psycop-t2d when reaching
maturity.
"""

from pathlib import Path

import wandb

import psycop_feature_generation.loaders.raw  # noqa
from application.t2d.modules.describe_flattened_dataset import (
    save_feature_set_description_to_disk,
)
from application.t2d.modules.flatten_dataset import create_flattened_dataset
from application.t2d.modules.project_setup import get_project_info, init_wandb
from application.t2d.modules.save_dataset_to_disk import split_and_save_dataset_to_disk
from application.t2d.modules.specify_features import get_feature_specs
from psycop_feature_generation.loaders.raw.load_demographic import birthdays
from psycop_feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)
from psycop_feature_generation.utils import FEATURE_SETS_PATH


def main(
    proj_name: str,
):
    """Main function for loading, generating and evaluating a flattened
    dataset.

    Args:
        proj_name (str): Name of project.
        feature_sets_path (Path): Path to where feature sets should be stored.
    """
    feature_specs = get_feature_specs()

    project_info = get_project_info(
        n_predictors=len(feature_specs.temporal_predictors),
        proj_name=proj_name,
    )

    init_wandb(
        wandb_project_name=project_info.project_name,
        predictor_specs=feature_specs.temporal_predictors,
        feature_set_path=project_info.feature_set_path,  # Save-dir as argument because we want to log the path
    )

    flattened_df = create_flattened_dataset(
        prediction_times=,
        spec_set=feature_specs,
        proj_path=proj_path,
        birthdays=birthdays(),
    )

    split_and_save_dataset_to_disk(
        flattened_df=flattened_df,
        out_dir=save_dir,
        file_prefix=feature_set_id,
        file_suffix="parquet",
    )

    save_feature_set_description_to_disk(
        predictor_specs=feature_specs.temporal_predictors
        + feature_specs.static_predictors,
        flattened_dataset_file_dir=save_dir,
        out_dir=save_dir,
        file_suffix="parquet",
    )

    wandb.log_artifact("poetry.lock", name="poetry_lock_file", type="poetry_lock")


if __name__ == "__main__":
    main(
        proj_name="t2d",
    )
