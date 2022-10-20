"""Example script to train multiple models and subsequently log the results to
wandb.

Usage:
- Replace the HYDRA_ARGS string with the desired arguments for `train_model.py`
- Run this script from project root with `python src/psycopt2d/train_and_log_models.py
"""
import os
from pathlib import Path

from random_word import RandomWords
from wasabi import msg

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

if __name__ == "__main__":
    time_spec = DatasetTimeSpecification(
        drop_patient_if_outcome_before_date=None,
        min_prediction_time_date="1979-01-01",
        min_lookbehind_days=0,
        min_lookahead_days=0,
    )

    dataset_spec = DatasetSpecification(
        file_suffix="csv",
        time=time_spec,
        pred_col_name_prefix="pred_",
        pred_time_colname="timestamp",
        split_dir_path=DATA_DIR,
    )

    loader = DataLoader(dataset_spec)
    train = loader.load_dataset_from_dir(split_names="train")

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

    # Override wandb group name with these
    # Generate random word-word string
    r = RandomWords()

    for lookbehind in possible_lookbehind_days:
        for lookahead in possible_lookahead_days:
            wandb_group = f"{r.get_random_word()}-{r.get_random_word()}-beh-{lookbehind}-ahead-{lookahead}"

            command = f"python src/psycopt2d/train_model.py {BASE_ARGS} data.min_lookbehind_days={lookbehind} data.min_lookahead_days={lookahead} +project.wandb_group={wandb_group} hydra.sweeper.n_trials={N_TRIALS_PER_CELL_IN_GRID} --config-name {BASE_CONF_FILE_NAME}"

            msg.info("Sending command")
            msg.info(command)

            os.system(command)
