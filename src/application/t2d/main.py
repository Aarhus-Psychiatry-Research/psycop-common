"""Main example on how to generate features.

Uses T2D-features. WIP, will be migrated to psycop-t2d when reaching
maturity.
"""

import wandb

import psycop_feature_generation.loaders.raw  # noqa pylint: disable=unused-import
from application.t2d.modules.describe_flattened_dataset import (
    save_flattened_dataset_description_to_disk,
)
from application.t2d.modules.flatten_dataset import create_flattened_dataset
from application.t2d.modules.project_setup import get_project_info, init_wandb
from application.t2d.modules.save_dataset_to_disk import split_and_save_dataset_to_disk
from application.t2d.modules.specify_features import get_feature_specs
from psycop_feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)


def main():
    """Main function for loading, generating and evaluating a flattened
    dataset.
    """
    project_info = get_project_info(
        project_name="t2d",
    )

    feature_specs = get_feature_specs(project_info=project_info)

    # Use wandb to keep track of your dataset generations
    # Makes it easier to find paths on wandb, as well as
    # allows monitoring and automatic slack alert on failure
    init_wandb(
        feature_specs=feature_specs,
        project_info=project_info,
    )

    flattened_df = create_flattened_dataset(
        feature_specs=feature_specs,
        prediction_times_df=physical_visits_to_psychiatry(),
        drop_pred_times_with_insufficient_look_distance=False,
        project_info=project_info,
    )

    split_and_save_dataset_to_disk(
        flattened_df=flattened_df,
        project_info=project_info,
        output_format="parquet",
    )

    save_flattened_dataset_description_to_disk(
        feature_specs=feature_specs,
        load_file_format="parquet",
        project_info=project_info,
    )

    wandb.log_artifact("poetry.lock", name="poetry_lock_file", type="poetry_lock")
