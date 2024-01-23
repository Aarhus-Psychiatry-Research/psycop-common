"""Main feature generation."""

import logging
import sys
from pathlib import Path

import wandb

from psycop.common.feature_generation.application_modules.describe_flattened_dataset import (
    save_flattened_dataset_description_to_disk,
)
from psycop.common.feature_generation.application_modules.flatten_dataset import (
    create_flattened_dataset,
)
from psycop.common.feature_generation.application_modules.loggers import init_root_logger
from psycop.common.feature_generation.application_modules.save_dataset_to_disk import (
    split_and_save_dataset_to_disk,
)
from psycop.common.feature_generation.application_modules.wandb_utils import (
    wandb_alert_on_exception,
)
from psycop.projects.restraint.feature_generation.modules.loaders.load_coercion_df_with_prediction_times_and_outcome import (
    load_coercion_prediction_times,
)
from psycop.projects.restraint.feature_generation.modules.specify_features import FeatureSpecifier
from psycop.projects.restraint.restraint_global_config import RESTRAINT_PROJECT_INFO

log = logging.getLogger()


@wandb_alert_on_exception
def main():
    """Main function for loading, generating and evaluating a flattened
    dataset."""
    project_info = RESTRAINT_PROJECT_INFO

    feature_specs = FeatureSpecifier(
        project_info=project_info,
        min_set_for_debug=False,  # Remember to set to False when generating full dataset
    ).get_feature_specs()

    flattened_df = create_flattened_dataset(
        feature_specs=feature_specs,  # type: ignore
        prediction_times_df=load_coercion_prediction_times(),
        drop_pred_times_with_insufficient_look_distance=False,
        project_info=project_info,
        add_birthdays=True,
    )

    split_and_save_dataset_to_disk(
        flattened_df=flattened_df,
        project_info=project_info,
        feature_set_dir=project_info.flattened_dataset_dir,
    )

    save_flattened_dataset_description_to_disk(
        project_info=project_info,
        feature_specs=feature_specs,  # type: ignore
        feature_set_dir=project_info.flattened_dataset_dir,
    )


if __name__ == "__main__":
    project_info = RESTRAINT_PROJECT_INFO

    init_root_logger(project_info=project_info)

    log.info(f"Stdout level is {logging.getLevelName(log.level)}")
    log.debug("Debugging is still captured in the log file")

    # Use wandb to keep track of your dataset generations
    # Makes it easier to find paths on wandb, as well as
    # allows monitoring and automatic slack alert on failure
    # allows monitoring and automatic slack alert on failure
    if sys.platform == "win32":
        (Path(__file__).resolve().parents[1] / "wandb" / "debug-cli.onerm").mkdir(
            exist_ok=True, parents=True
        )

    wandb.init(
        project=f"{project_info.project_name}-feature-generation",
        entity="psycop",
        config={"feature_set_path": project_info.flattened_dataset_dir},
        mode="offline",
    )

    main()
