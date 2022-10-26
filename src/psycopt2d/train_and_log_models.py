"""Example script to train (multiple) models and simultaneously log the results
to wandb. Logs AUC for all models, and runs the full model evaluation suite on
the best performing ones.

Usage:
- Set the constants in the first section of the script (mainly under the # RUN CONSTANTS header).
- Replace the HYDRA_ARGS string with the desired arguments for `train_model.py`
- Run this script from project root with `python src/psycopt2d/train_and_log_models.py`
"""

import subprocess
import time

from hydra import compose, initialize
from wasabi import Printer

# RUN CONSTANTS
CONFIG_NAME = "integration_testing.yaml"

HYDRA_ARGS = f"--multirun project.wandb.mode='dryrun' model.args.tree_method='auto' --config-name {CONFIG_NAME}"
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

if __name__ == "__main__":
    msg = Printer(timestamp=True)

    with initialize(version_base=None, config_path="config/"):
        cfg = compose(
            config_name=CONFIG_NAME,
        )
    trainer = subprocess.Popen(
        ["python", "src/psycopt2d/train_model.py", *HYDRA_ARGS.split(" ")],
    )
    watcher = subprocess.Popen(
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
    while trainer.poll() is None:
        time.sleep(1)

    msg.good(
        f"Training finished. Stopping the watcher in {KEEP_WATCHER_ALIVE_AFTER_TRAINING_FINISHED_MINUTES} minutes...",
    )
    time.sleep(60 * KEEP_WATCHER_ALIVE_AFTER_TRAINING_FINISHED_MINUTES)
    watcher.kill()
    msg.good("Watcher stopped.")
