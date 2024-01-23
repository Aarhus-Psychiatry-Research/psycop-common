import logging
from pathlib import Path

import pandas as pd
from timeseriesflattener.feature_specs.single_specs import AnySpec

from psycop.common.feature_generation.application_modules.chunked_feature_generation import (
    ChunkedFeatureGenerator,
)
from psycop.common.feature_generation.application_modules.describe_flattened_dataset import (
    save_flattened_dataset_description_to_disk,
)
from psycop.common.feature_generation.application_modules.flatten_dataset import (
    create_flattened_dataset,
)
from psycop.common.feature_generation.application_modules.loggers import init_root_logger
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
def generate_feature_set(
    project_info: ProjectInfo,
    eligible_prediction_times: pd.DataFrame,
    feature_specs: list[AnySpec],
    generate_in_chunks: bool = False,
    chunksize: int = 250,
    feature_set_name: str | None = None,
) -> Path:
    """Main function for loading, generating and evaluating a flattened
    dataset.
    If generate_in_chunks is True, feature generation is split into
    multiple chunks to avoid memory issues"""

    if feature_set_name:
        feature_set_dir = project_info.flattened_dataset_dir / feature_set_name
    else:
        feature_set_dir = project_info.flattened_dataset_dir

    if Path.exists(feature_set_dir):
        while True:
            response = input(
                f"The path '{feature_set_dir}' already exists. Do you want to potentially overwrite the contents of this folder with new feature sets? (yes/no): "
            )

            if response.lower() not in ["yes", "y", "no", "n"]:
                print("Invalid response. Please enter 'yes/y' or 'no/n'.")
            if response.lower() in ["no", "n"]:
                print("Process stopped.")
                return feature_set_dir
            if response.lower() in ["yes", "y"]:
                print(f"Folder '{feature_set_dir}' will be overwritten.")
                break

    if generate_in_chunks:
        flattened_df = ChunkedFeatureGenerator.create_flattened_dataset_with_chunking(
            project_info, eligible_prediction_times, feature_specs, chunksize
        )

    else:
        flattened_df = create_flattened_dataset(
            feature_specs=feature_specs,
            prediction_times_df=eligible_prediction_times,
            drop_pred_times_with_insufficient_look_distance=False,
            project_info=project_info,
        )

    split_and_save_dataset_to_disk(
        flattened_df=flattened_df, project_info=project_info, feature_set_dir=feature_set_dir
    )

    save_flattened_dataset_description_to_disk(
        project_info=project_info,
        feature_specs=feature_specs,  # type: ignore
        feature_set_dir=feature_set_dir,
    )

    return feature_set_dir


def init_wandb_and_generate_feature_set(
    project_info: ProjectInfo,
    eligible_prediction_times: pd.DataFrame,
    feature_specs: list[AnySpec],
    generate_in_chunks: bool = False,
    chunksize: int = 250,
    feature_set_name: str | None = None,
) -> Path:
    # Run elements that are required before wandb init first,
    # then run the rest in main so you can wrap it all in
    # wandb_alert_on_exception, which will send a slack alert
    # if you have wandb alerts set up in wandb
    init_logger_and_wandb(project_info)

    return generate_feature_set(
        project_info=project_info,
        eligible_prediction_times=eligible_prediction_times,
        feature_specs=feature_specs,
        generate_in_chunks=generate_in_chunks,
        chunksize=chunksize,
        feature_set_name=feature_set_name,
    )


def init_logger_and_wandb(project_info: ProjectInfo):
    init_root_logger(project_info=project_info)

    log.info(  # pylint: disable=logging-fstring-interpolation
        f"Stdout level is {logging.getLevelName(log.level)}"
    )
    log.debug("Debugging is still captured in the log file")

    # Use wandb to keep track of your dataset generations
    # Makes it easier to find paths on wandb, as well as
    # allows monitoring and automatic slack alert on failure
    init_wandb(project_info=project_info)  # allows monitoring and automatic slack alert on failure
