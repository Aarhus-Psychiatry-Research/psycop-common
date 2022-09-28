"""Example script to train multiple models and subsequently log the results to
wandb.

Usage:
- Replace the HYDRA_ARGS string with the desired arguments for `train_model.py`
- Run this script from project root with `python src/psycopt2d/train_and_log_models.py`
-
"""
import os

from psycopt2d.utils import AUC_LOGGING_FILE_PATH

HYDRA_ARGS = "--multirun +model=xgboost --config-name integration_testing.yaml"
WANDB_PROJECT = "psycopt2d-testing"
SAVE_TOP_N = 5

if __name__ == "__main__":
    # Remove previously logged runs
    if AUC_LOGGING_FILE_PATH.exists():
        AUC_LOGGING_FILE_PATH.unlink()
    # train models
    os.system(f"python src/psycopt2d/train_model.py {HYDRA_ARGS}")
    # sync to wandb and move runs to archive
    os.system(
        f"python src/psycopt2d/sync_best_runs_to_wandb.py --top_n {SAVE_TOP_N} --project {WANDB_PROJECT}",
    )
