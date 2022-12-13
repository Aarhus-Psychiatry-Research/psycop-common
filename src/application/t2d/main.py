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
    project_name: str,
):
    """Main function for loading, generating and evaluating a flattened
    dataset.

    Args:
        project_name (str): Name of project.
    """
    feature_specs = get_feature_specs()

    project_info = get_project_info(
        n_predictors=len(feature_specs.temporal_predictors),
        project_name=project_name,
    )

    # Use wandb to keep track of your dataset generations
    # Makes it easier to find paths on wandb, as well as
    # allows monitoring and automatic slack alert on failure
    init_wandb(
        wandb_project_name=project_info.project_name,
        predictor_specs=feature_specs.temporal_predictors,
        feature_set_path=project_info.feature_set_path,  # Save-dir as argument because we want to log the path
    )

    flattened_df = create_flattened_dataset(
        feature_specs=feature_specs,
        project_info=project_info,
    )

    split_and_save_dataset_to_disk(
        flattened_df=flattened_df,
        project_info=project_info,
        output_format="parquet",
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
        project_name="t2d",
    )
