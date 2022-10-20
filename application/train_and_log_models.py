"""Example script to train multiple models and subsequently log the results to
wandb.

Usage:
- Replace the HYDRA_ARGS string with the desired arguments for `train_model.py`
- Run this script from project root with `python src/psycopt2d/train_and_log_models.py
"""
import os
import subprocess
import time
from pathlib import Path
from typing import Iterable, Union

from hydra import compose, initialize
from pydantic import BaseModel
from wasabi import Printer, msg

from psycopt2d.evaluate_saved_model_predictions import (
    infer_look_distance,
    infer_outcome_col_name,
    infer_predictor_col_name,
)
from psycopt2d.load import DataLoader, DatasetSpecification, DatasetTimeSpecification

BASE_CONF_FILE_NAME = "integration_testing.yaml"

DATA_DIR = Path("/Users/au484925/Desktop/psycop-t2d/tests/test_data/synth_splits/")

BASE_ARGS = "--multirun +model=xgboost"
WANDB_PROJECT = "psycopt2d-testing"
N_TRIALS_PER_CELL_IN_GRID = 50

# RUN CONSTANTS
CONFIG_NAME = "integration_testing.yaml"

HYDRA_ARGS = f"--multirun +model=xgboost project.wandb_mode='dryrun' model.args.tree_method='auto' --config-name {CONFIG_NAME}"
OVERTACI = "false"  # Change to "true" if running on overtaci

# WATCHER CONSTANTS
WANDB_ENTITY = (
    "psycop"  # The wandb entity to upload to (e.g. "psycop" or your user name)
)
N_RUNS_BEFORE_FIRST_EVAL = (
    "1"  # The number of runs to upload to wandb before evaluating the best runs.
)
KEEP_WATCHER_ALIVE_AFTER_TRAINING_FINISHED_MINUTES = (
    5  # minutes to wait for the wandb watcher after training
)
# has finished. Will kill the watcher after this time.
ARCHIVE_ALL_WANDB_RUNS = "false"  # whether to archive all runs in the wandb folder
# before starting model training. Change to "t" to archive all wandb runs


def load_data(dataset_spec):
    """Load the data"""
    loader = DataLoader(dataset_spec)
    return loader.load_dataset_from_dir(split_names="train")


class PossibleLookDistanceDays(BaseModel):
    ahead: Iterable[Union[int, float]]
    behind: Iterable[Union[int, float]]


def infer_possible_look_directions(train):
    """Infer the possible values for min_lookahead_days and min_lookbehind_days"""
    # Get potential lookaheads from outc_ columns
    outcome_col_names = infer_outcome_col_name(df=train, allow_multiple=True)

    possible_lookahead_days = set(
        infer_look_distance(
            col_name=outcome_col_names,
        ),
    )

    # Get potential lookbehinds from pred_ columns
    pred_col_names = infer_predictor_col_name(df=train, allow_multiple=True)
    possible_lookbehind_days = set(infer_look_distance(col_name=pred_col_names))

    return PossibleLookDistanceDays(
        ahead=possible_lookahead_days, behind=possible_lookbehind_days
    )


def get_dataset_spec(data_dir_path: Path):
    time_spec = DatasetTimeSpecification(
        drop_patient_if_outcome_before_date=None,
        min_prediction_time_date="1979-01-01",
        min_lookbehind_days=0,
        min_lookahead_days=0,
    )

    return DatasetSpecification(
        file_suffix="csv",
        time=time_spec,
        pred_col_name_prefix="pred_",
        pred_time_colname="timestamp",
        split_dir_path=data_dir_path,
    )


def train_models_for_each_grid(
    base_conf_file_name: Union[str, Path],
    base_args: str,
    n_trials_per_cell_in_grid: int,
    possible_look_distances: PossibleLookDistanceDays,
):
    """Train a model for each cell in the grid of possible look distances"""
    from random_word import RandomWords

    random_word = RandomWords()

    for lookbehind in possible_look_distances.behind:
        for lookahead in possible_look_distances.ahead:
            wandb_group = f"{random_word.get_random_word()}-{random_word.get_random_word()}-beh-{lookbehind}-ahead-{lookahead}"

            command = f"python src/psycopt2d/train_model.py {base_args} data.min_lookbehind_days={lookbehind} data.min_lookahead_days={lookahead} +project.wandb_group={wandb_group} hydra.sweeper.n_trials={n_trials_per_cell_in_grid} --config-name {base_conf_file_name}"

            msg.info("Sending command")
            msg.info(command)

            os.system(command)


if __name__ == "__main__":
    msg = Printer(timestamp=True)

    with initialize(version_base=None, config_path="config/"):
        cfg = compose(
            config_name=CONFIG_NAME,
        )

    dataset_spec = get_dataset_spec(data_dir_path=DATA_DIR)

    train = load_data(dataset_spec=dataset_spec)

    possible_look_distance = infer_possible_look_directions(train)

    watcher = subprocess.Popen(  # pylint: disable=consider-using-with
        [
            "python",
            "src/psycopt2d/model_training_watcher.py",
            "--entity",
            WANDB_ENTITY,
            "--project_name",
            cfg.project.name,
            "--n_runs_before_eval",
            N_RUNS_BEFORE_FIRST_EVAL,
            "--overtaci",
            OVERTACI,
            "--timeout",
            "None",
            "--clean_wandb_dir",
            ARCHIVE_ALL_WANDB_RUNS,
        ],
    )

    train_models_for_each_grid(
        base_conf_file_name=BASE_CONF_FILE_NAME,
        base_args=BASE_ARGS,
        n_trials_per_cell_in_grid=N_TRIALS_PER_CELL_IN_GRID,
        possible_look_distances=possible_look_distance,
    )

    msg.good(
        f"Training finished. Stopping the watcher in {KEEP_WATCHER_ALIVE_AFTER_TRAINING_FINISHED_MINUTES} minutes...",
    )

    time.sleep(60 * KEEP_WATCHER_ALIVE_AFTER_TRAINING_FINISHED_MINUTES)
    watcher.kill()
    msg.good("Watcher stopped.")
