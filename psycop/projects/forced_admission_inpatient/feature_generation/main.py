"""Main feature generation."""

import logging
import sys
import warnings
from pathlib import Path
from typing import Literal

from psycop.common.feature_generation.application_modules.chunked_feature_generation import (
    ChunkedFeatureGenerator,
)
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
from psycop.projects.forced_admission_inpatient.cohort.forced_admissions_inpatient_cohort_definition import (
    ForcedAdmissionsInpatientCohortDefiner,
)
from psycop.projects.forced_admission_inpatient.feature_generation.modules.specify_features import (
    FeatureSpecifier,
)
from psycop.projects.forced_admission_inpatient.feature_generation.modules.specify_text_features import (
    TextFeatureSpecifier,
)

log = logging.getLogger()
warnings.simplefilter(action="ignore", category=RuntimeWarning)


@wandb_alert_on_exception
def main(
    add_text_features: bool = True,
    min_set_for_debug: bool = False,
    limited_feature_set: bool = False,
    lookbehind_180d_mean: bool = False,
    generate_in_chunks: bool = True,
    washout_on_prior_forced_admissions: bool = True,
    feature_set_name: str | None = None,
    text_embedding_method: Literal["tfidf", "sentence_transformer", "both"] = "both",
    chunksize: int = 10,
) -> Path:
    """Main function for loading, generating and evaluating a flattened
    dataset."""

    if feature_set_name:
        feature_set_dir = project_info.flattened_dataset_dir / feature_set_name
    else:
        feature_set_dir = project_info.flattened_dataset_dir

    if Path.exists(feature_set_dir):
        while True:
            response = input(
                f"The path '{feature_set_dir}' already exists. Do you want to potentially overwrite the contents of this folder with new feature sets? (yes/no): ",
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
        limited_feature_set=limited_feature_set,
        lookbehind_180d_mean=lookbehind_180d_mean,
    ).get_feature_specs()

    if add_text_features:
        text_feature_specs = TextFeatureSpecifier(
            project_info=project_info,
            min_set_for_debug=min_set_for_debug,  # Remember to set to False when generating full dataset
        ).get_text_feature_specs(
            embedding_method=text_embedding_method,
        )  # type: ignore

        feature_specs += text_feature_specs

    if generate_in_chunks:
        flattened_df = ChunkedFeatureGenerator.create_flattened_dataset_with_chunking(
            project_info=project_info,
            eligible_prediction_times=ForcedAdmissionsInpatientCohortDefiner.get_filtered_prediction_times_bundle(
                washout_on_prior_forced_admissions=washout_on_prior_forced_admissions,
            ).prediction_times.to_pandas(),
            feature_specs=feature_specs,  # type: ignore
            chunksize=chunksize,
        )

    else:
        flattened_df = create_flattened_dataset(
            feature_specs=feature_specs,  # type: ignore
            prediction_times_df=ForcedAdmissionsInpatientCohortDefiner.get_filtered_prediction_times_bundle(
                washout_on_prior_forced_admissions=washout_on_prior_forced_admissions,
            ).prediction_times.to_pandas(),
            drop_pred_times_with_insufficient_look_distance=False,
            project_info=project_info,
        )

    split_and_save_dataset_to_disk(
        flattened_df=flattened_df,
        project_info=project_info,
        feature_set_dir=feature_set_dir,
    )

    save_flattened_dataset_description_to_disk(
        project_info=project_info,
        feature_specs=feature_specs,  # type: ignore
        feature_set_dir=feature_set_dir,
    )
    return feature_set_dir


if __name__ == "__main__":
    # Run elements that are required before wandb init first,
    # then run the rest in main so you can wrap it all in
    # wandb_alert_on_exception, which will send a slack alert
    # if you have wandb alerts set up in wandb
    project_info = ProjectInfo(
        project_name="forced_admissions_inpatient",
        project_path=OVARTACI_SHARED_DIR / "forced_admissions_inpatient",
    )

    init_root_logger(project_info=project_info)

    log.info(
        f"Stdout level is {logging.getLevelName(log.level)}",
    )  # pylint: disable=logging-fstring-interpolation
    log.debug("Debugging is still captured in the log file")

    # Use wandb to keep track of your dataset generations
    # Makes it easier to find paths on wandb, as well as
    # allows monitoring and automatic slack alert on failure
    # allows monitoring and automatic slack alert on failure
    if sys.platform == "win32":
        (Path(__file__).resolve().parents[0] / "wandb" / "debug-cli.onerm").mkdir(
            exist_ok=True,
            parents=True,
        )

    init_wandb(
        project_info=project_info,
    )

    main(
        add_text_features=False,
        min_set_for_debug=False,
        limited_feature_set=True,
        lookbehind_180d_mean=False,
        washout_on_prior_forced_admissions=True,
        feature_set_name="limited_features_set_demographics_diagnoses",
        generate_in_chunks=False,
    )
