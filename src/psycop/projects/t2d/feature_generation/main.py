"""Main feature generation."""

import logging
from pathlib import Path

from psycop.common.feature_generation.application_modules.describe_flattened_dataset import (
    save_flattened_dataset_description_to_disk,
)
from psycop.common.feature_generation.application_modules.flatten_dataset import (
    create_flattened_dataset,
)
from psycop.common.feature_generation.application_modules.loggers import (
    init_root_logger,
)
from psycop.common.feature_generation.application_modules.project_setup import (
    ProjectInfo,
    init_wandb,
)
from psycop.common.feature_generation.application_modules.save_dataset_to_disk import (
    split_and_save_dataset_to_disk,
)
from psycop.common.feature_generation.application_modules.wandb_utils import (
    wandb_alert_on_exception,
)
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.projects.t2d.feature_generation.eligible_prediction_times.loader import (
    get_eligible_prediction_times_as_pandas,
)
from psycop.projects.t2d.feature_generation.specify_features import FeatureSpecifier

log = logging.getLogger()


@wandb_alert_on_exception
def _generate_feature_set(project_info: ProjectInfo) -> Path:
    """Main function for loading, generating and evaluating a flattened
    dataset."""
    feature_specs = FeatureSpecifier(
        project_info=project_info,
        min_set_for_debug=False,  # Remember to set to False when generating full dataset
    ).get_feature_specs()

    flattened_df = create_flattened_dataset(
        feature_specs=feature_specs,
        prediction_times_df=get_eligible_prediction_times_as_pandas(),
        drop_pred_times_with_insufficient_look_distance=False,
        project_info=project_info,
    )

    split_and_save_dataset_to_disk(
        flattened_df=flattened_df,
        project_info=project_info,
    )

    save_flattened_dataset_description_to_disk(
        project_info=project_info,
        feature_specs=feature_specs,  # type: ignore
    )

    return project_info.flattened_dataset_dir


def generate_feature_set() -> Path:
    # Run elements that are required before wandb init first,
    # then run the rest in main so you can wrap it all in
    # wandb_alert_on_exception, which will send a slack alert
    # if you have wandb alerts set up in wandb
    project_name = "t2d"

    project_info = ProjectInfo(
        project_name=project_name,
        project_path=OVARTACI_SHARED_DIR / project_name,
    )

    init_root_logger(project_info=project_info)

    log.info(  # pylint: disable=logging-fstring-interpolation
        f"Stdout level is {logging.getLevelName(log.level)}",
    )
    log.debug("Debugging is still captured in the log file")

    # Use wandb to keep track of your dataset generations
    # Makes it easier to find paths on wandb, as well as
    # allows monitoring and automatic slack alert on failure
    init_wandb(
        project_info=project_info,
    )

    return _generate_feature_set(project_info=project_info)


if __name__ == "__main__":
    generate_feature_set()
