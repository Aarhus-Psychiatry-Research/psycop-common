"""Main feature generation."""

import logging
import sys
from pathlib import Path

from psycop.common.feature_generation.application_modules.chunked_feature_generation import (
    ChunkedFeatureGenerator,
)
from psycop.common.feature_generation.application_modules.describe_flattened_dataset import (
    save_flattened_dataset_description_to_disk,
)
from psycop.common.feature_generation.application_modules.flatten_dataset import (
    create_flattened_dataset_tsflattener_v1,
)
from psycop.common.feature_generation.application_modules.loggers import init_root_logger
from psycop.common.feature_generation.application_modules.save_dataset_to_disk import (
    split_and_save_dataset_to_disk,
)
from psycop.projects.restraint.feature_generation.modules.loaders.load_restraint_prediction_timestamps import (
    load_restraint_prediction_timestamps,
)
from psycop.projects.restraint.feature_generation.modules.specify_features import FeatureSpecifier
from psycop.projects.restraint.feature_generation.modules.specify_text_features import (
    TextFeatureSpecifier,
)
from psycop.projects.restraint.restraint_global_config import RESTRAINT_PROJECT_INFO

log = logging.getLogger()


def main(
    add_text_features: bool = True,
    generate_in_chunks: bool = True,
    min_set_for_debug: bool = False,
    feature_set_name: str | None = None,
    chunksize: int = 10,
) -> None | Path:
    """Main function for loading, generating and evaluating a flattened
    dataset."""
    project_info = RESTRAINT_PROJECT_INFO

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

    feature_specs = FeatureSpecifier(
        project_info=project_info,
        min_set_for_debug=min_set_for_debug,  # Remember to set to False when generating full dataset
    ).get_feature_specs()

    if add_text_features:
        text_feature_specs = TextFeatureSpecifier(
            project_info=project_info,
            min_set_for_debug=min_set_for_debug,  # Remember to set to False when generating full dataset
        ).get_text_feature_specs()  # type: ignore

        feature_specs += text_feature_specs

    if generate_in_chunks:
        flattened_df = ChunkedFeatureGenerator.create_flattened_dataset_with_chunking(
            project_info=project_info,
            eligible_prediction_times=load_restraint_prediction_timestamps()[
                ["dw_ek_borger", "timestamp"]
            ],
            feature_set_dir=feature_set_dir,
            feature_specs=feature_specs,  # type: ignore
            chunksize=chunksize,
        )

    flattened_df = create_flattened_dataset_tsflattener_v1(
        feature_specs=feature_specs,  # type: ignore
        prediction_times_df=load_restraint_prediction_timestamps()[["dw_ek_borger", "timestamp"]],
        drop_pred_times_with_insufficient_look_distance=True,
        project_info=project_info,
        add_birthdays=True,
    )

    split_and_save_dataset_to_disk(
        flattened_df=flattened_df, project_info=project_info, feature_set_dir=feature_set_dir
    )

    save_flattened_dataset_description_to_disk(
        project_info=project_info,
        feature_specs=feature_specs,  # type: ignore
        feature_set_dir=feature_set_dir,
    )
    return None


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

    main(
        add_text_features=True,
        min_set_for_debug=False,
        feature_set_name="full_feature_set_structured_tfidf_750_all_outcomes",
        generate_in_chunks=True,
    )
