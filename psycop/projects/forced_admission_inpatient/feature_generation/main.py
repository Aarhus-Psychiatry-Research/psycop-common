"""Main feature generation."""

import logging
import sys
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
from psycop.common.feature_generation.loaders.raw.load_moves import (
    load_move_into_rm_for_exclusion,
)
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.projects.forced_admission_inpatient.feature_generation.modules.loaders.load_forced_admissions_dfs_with_prediction_times_and_outcome import (
    forced_admissions_inpatient,
)
from psycop.projects.forced_admission_inpatient.feature_generation.modules.specify_features import (
    FeatureSpecifier,
)
from psycop.projects.forced_admission_inpatient.feature_generation.modules.utils import (
    add_outcome_col,
)

log = logging.getLogger()


@wandb_alert_on_exception
def main():
    """Main function for loading, generating and evaluating a flattened
    dataset."""
    feature_specs = FeatureSpecifier(
        project_info=project_info,
        min_set_for_debug=False,  # Remember to set to False when generating full dataset
        limited_feature_set=True,
    ).get_feature_specs()

    flattened_df = create_flattened_dataset(
        feature_specs=feature_specs,  # type: ignore
        prediction_times_df=forced_admissions_inpatient(
            timestamps_only=True,
        ),
        drop_pred_times_with_insufficient_look_distance=False,
        project_info=project_info,
        quarantine_df=load_move_into_rm_for_exclusion(),
        quarantine_days=720,
    )

    flattened_df = add_outcome_col(
        flattened_df=flattened_df,
        visit_type="inpatient",
    )

    split_and_save_dataset_to_disk(
        flattened_df=flattened_df,
        project_info=project_info,
    )

    save_flattened_dataset_description_to_disk(
        feature_specs=feature_specs,  # type: ignore
        project_info=project_info,
    )


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

    main()
