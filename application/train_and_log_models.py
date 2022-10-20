"""Example script to train multiple models and subsequently log the results to
wandb.

Usage:
- Replace the HYDRA_ARGS string with the desired arguments for `train_model.py`
- Run this script from project root with `python src/psycopt2d/train_and_log_models.py`
-
"""
import os
from pathlib import Path

from psycopt2d.evaluate_saved_model_predictions import (
    infer_look_distance,
    infer_outcome_col_name,
    infer_predictor_col_names,
)
from psycopt2d.load import DataLoader, DatasetSpecification, DatasetTimeSpecification

BASE_CONF_FILE_NAME = f"default_config.yaml"
DATA_DIR = (
    Path("E:")
    / "shared_resources"
    / "feature_sets"
    / "t2d"
    / "feature_sets"
    / "psycop_t2d_adminmanber_201_features_2022_10_05_15_14"
)

BASE_ARGS = f"--multirun +model=xgboost --config-name {BASE_CONF_FILE_NAME}"
WANDB_PROJECT = "psycopt2d-testing"

if __name__ == "__main__":
    time_spec = DatasetTimeSpecification(
        drop_patient_if_outcome_before_date="1979-01-01",
        min_prediction_time_date="1979-01-01",
        min_lookbehind_days=0,
        min_lookahead_days=0,
    )

    dataset_spec = DatasetSpecification(
        file_suffix="parquet",
        time_spec=time_spec,
        pred_col_name_prefix="pred_",
        pred_time_colname="timestamp",
        split_dir_path=DATA_DIR,
        time=time_spec,
    )

    loader = DataLoader(dataset_spec)
    train = loader.load_dataset_from_dir(split_names="train")

    # Get potential lookaheads from outc_ columns
    outcome_col_names = infer_outcome_col_name(df=train, allow_multiple=True)
    possible_lookahead_days = infer_look_distance(
        col_names=outcome_col_names, allow_multiple=True
    )

    # Get potential lookbehinds from pred_ columns
    pred_col_names = infer_predictor_col_names(df=train, allow_multiple=True)
    possible_lookbehind_days = infer_look_distance(col_names=pred_col_names)

    # Override wandb group name with these

    # Iterate over them

    # Add feature subsetting subsetting to args
    os.system(f"python src/psycopt2d/train_model.py {BASE_ARGS} ")
