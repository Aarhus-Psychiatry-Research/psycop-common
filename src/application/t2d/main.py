"""Main example on how to generate features.

Uses T2D-features. WIP, will be migrated to psycop-t2d when reaching
maturity.
"""

import logging

import wandb

from application.t2d.modules.specify_features import FeatureSpecifier
from psycop_feature_generation.application_modules.describe_flattened_dataset import (
    save_flattened_dataset_description_to_disk,
)
from psycop_feature_generation.application_modules.flatten_dataset import (
    create_flattened_dataset,
)
from psycop_feature_generation.application_modules.loggers import init_root_logger
from psycop_feature_generation.application_modules.project_setup import (
    get_project_info,
    init_wandb,
)
from psycop_feature_generation.application_modules.save_dataset_to_disk import (
    split_and_save_dataset_to_disk,
)
from psycop_feature_generation.application_modules.wandb_utils import (
    wandb_alert_on_exception,
)
from psycop_feature_generation.loaders.raw.load_moves import (
    load_move_into_rm_for_exclusion,
)
from psycop_feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)

log = logging.getLogger()


@wandb_alert_on_exception
def main():
    """Main function for loading, generating and evaluating a flattened
    dataset."""
    feature_specs = FeatureSpecifier(
        project_info=project_info, min_set_for_debug=True
    ).get_feature_specs()

    flattened_df = create_flattened_dataset(
        feature_specs=feature_specs,
        prediction_times_df=physical_visits_to_psychiatry(timestamps_only=True),
        drop_pred_times_with_insufficient_look_distance=False,
        project_info=project_info,
        quarantine_df=load_move_into_rm_for_exclusion(),
        quarantine_days=720,
    )

    split_and_save_dataset_to_disk(
        flattened_df=flattened_df,
        project_info=project_info,
    )

    save_flattened_dataset_description_to_disk(
        project_info=project_info,
    )

    wandb.log_artifact("poetry.lock", name="poetry_lock_file", type="poetry_lock")


if __name__ == "__main__":
    # Run elements that are required before wandb init first,
    # then run the rest in main so you can wrap it all in
    # wandb_alert_on_exception, which will send a slack alert
    # if you have wandb alerts set up in wandb
    project_info = get_project_info(
        project_name="t2d",
    )

    init_root_logger(project_info=project_info)

    log.info(f"Stdout level is {logging.getLevelName(log.level)}")
    log.debug("Debugging is still captured in the log file")

    # Use wandb to keep track of your dataset generations
    # Makes it easier to find paths on wandb, as well as
    # allows monitoring and automatic slack alert on failure
    init_wandb(
        project_info=project_info,
    )

    main()
