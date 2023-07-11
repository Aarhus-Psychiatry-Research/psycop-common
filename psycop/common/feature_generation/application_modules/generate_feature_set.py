import logging
from pathlib import Path

import pandas as pd
from timeseriesflattener.feature_specs.single_specs import AnySpec

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

log = logging.getLogger()


@wandb_alert_on_exception
def _generate_feature_set(
    project_info: ProjectInfo,
    eligible_prediction_times: pd.DataFrame,
    feature_specs: list[AnySpec],
) -> Path:
    """Main function for loading, generating and evaluating a flattened
    dataset."""
    flattened_df = create_flattened_dataset(
        feature_specs=feature_specs,
        prediction_times_df=eligible_prediction_times,
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


def init_wandb_and_generate_feature_set(
    project_info: ProjectInfo,
    eligible_prediction_times: pd.DataFrame,
    feature_specs: list[AnySpec],
) -> Path:
    # Run elements that are required before wandb init first,
    # then run the rest in main so you can wrap it all in
    # wandb_alert_on_exception, which will send a slack alert
    # if you have wandb alerts set up in wandb
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

    return _generate_feature_set(
        project_info=project_info,
        eligible_prediction_times=eligible_prediction_times,
        feature_specs=feature_specs,
    )  # allows monitoring and automatic slack alert on failure
