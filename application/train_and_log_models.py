"""Example script to train multiple models and subsequently log the results to
wandb.

Usage:
- Replace the HYDRA_ARGS string with the desired arguments for `train_model.py`
- Run this script from project root with `python src/psycopt2d/train_and_log_models.py
"""
import random
import subprocess
import time

from hydra import compose, initialize
from pydantic import BaseModel
from wasabi import Printer

from psycopt2d.evaluate_saved_model_predictions import (
    infer_look_distance,
    infer_outcome_col_name,
    infer_predictor_col_name,
)
from psycopt2d.load import DataLoader
from psycopt2d.utils.omegaconf_to_pydantic_objects import (
    FullConfig,
    omegaconf_to_pydantic_objects,
)

msg = Printer(timestamp=True)


class PossibleLookDistanceDays(BaseModel):
    """Possible look distances."""

    ahead: list[str]
    behind: list[str]


def load_train_for_inference(cfg: FullConfig):
    """Load the data."""
    loader = DataLoader(cfg=cfg)
    msg.info("Loading datasets for look direction inference")
    return loader.load_dataset_from_dir(split_names="train")


def infer_possible_look_directions(train):
    """Infer the possible values for min_lookahead_days and
    min_lookbehind_days."""
    # Get potential lookaheads from outc_ columns
    outcome_col_names = infer_outcome_col_name(df=train, allow_multiple=True)

    possible_lookahead_days = infer_look_distance(col_name=outcome_col_names)

    # Get potential lookbehinds from pred_ columns
    pred_col_names = infer_predictor_col_name(df=train, allow_multiple=True)
    possible_lookbehind_days = list(set(infer_look_distance(col_name=pred_col_names)))

    return PossibleLookDistanceDays(
        ahead=possible_lookahead_days,
        behind=possible_lookbehind_days,
    )


class LookDirectionCombination(BaseModel):
    """A combination of lookbehind and lookahead days."""

    lookbehind: int
    lookahead: int


def train_models_for_each_cell_in_grid(
    cfg: FullConfig,
    possible_look_distances: PossibleLookDistanceDays,
    config_file_name: str,
):
    """Train a model for each cell in the grid of possible look distances."""
    from random_word import RandomWords

    random_word = RandomWords()

    # Create all combinations of lookbehind and lookahead days
    lookbehind_combinations = [
        LookDirectionCombination(lookbehind=lookbehind, lookahead=lookahead)
        for lookbehind in possible_look_distances.behind
        for lookahead in possible_look_distances.ahead
    ]

    lookbehind_combinations = [
        comb for comb in lookbehind_combinations if comb.lookahead <= 1095
    ]

    random.shuffle(lookbehind_combinations)

    active_trainers: list[subprocess.Popen] = []

    wandb_prefix = f"{random_word.get_random_word()}-{random_word.get_random_word()}"

    while lookbehind_combinations:
        # Loop to run if enough trainers have been spawned
        if len(active_trainers) >= 4:
            active_trainers = [t for t in active_trainers if t.poll() is None]
            time.sleep(1)
            continue

        cell = lookbehind_combinations.pop()
        msg.info(
            f"Spawning a new trainer with lookbehind={cell.lookbehind} and lookahead={cell.lookahead}",
        )

        wandb_group = f"{wandb_prefix}-beh-{cell.lookbehind}-ahead-{cell.lookahead}"

        subprocess_args: list[str] = [
            "python",
            "src/psycopt2d/train_model.py",
            f"model={cfg.model.model_name}",
            f"data.min_lookbehind_days={cell.lookbehind}",
            f"data.min_lookahead_days={cell.lookahead}",
            f"project.wandb_group='{wandb_group}'",
            f"hydra.sweeper.n_trials={cfg.train.n_trials_per_lookdirection_combination}",
            f"project.wandb_mode={cfg.project.wandb_mode}",
            "--config-name",
            f"{config_file_name}",
        ]

        if cfg.train.n_trials_per_lookdirection_combination > 1:
            subprocess_args.insert(2, "--multirun")

        if cfg.model.model_name == "xgboost" and not cfg.project.gpu:
            subprocess_args.insert(3, "++model.args.tree_method='auto'")

        msg.info(f'{" ".join(subprocess_args)}')

        active_trainers.append(
            subprocess.Popen(  # pylint: disable=consider-using-with
                args=subprocess_args,
            ),
        )


if __name__ == "__main__":
    msg = Printer(timestamp=True)

    CONFIG_FILE_NAME = "default_config.yaml"

    with initialize(version_base=None, config_path="../src/psycopt2d/config/"):
        cfg = compose(
            config_name=CONFIG_FILE_NAME,
        )

        cfg = omegaconf_to_pydantic_objects(cfg)

    watcher = subprocess.Popen(  # pylint: disable=consider-using-with
        [
            "python",
            "src/psycopt2d/model_training_watcher.py",
            "--entity",
            cfg.project.wandb_entity,
            "--project_name",
            cfg.project.name,
            "--n_runs_before_eval",
            str(cfg.project.watcher.n_runs_before_eval),
            "--overtaci",
            cfg.eval.save_model_predictions_on_overtaci,
            "--timeout",
            "None",
            "--clean_wandb_dir",
            cfg.project.watcher.archive_all,
        ],
    )

    train = load_train_for_inference(cfg=cfg)

    possible_look_distances = infer_possible_look_directions(train)

    # Remove "9999" from possible look distances behind
    possible_look_distances.behind = [
        dist for dist in possible_look_distances.behind if dist != "9999"
    ]

    msg.info(f"Possible lookbehind days: {possible_look_distances.behind}")
    msg.info(f"Possible lookahead days: {possible_look_distances.ahead}")

    if not cfg.project.gpu:
        msg.warn("Not using GPU for training")

    CLEAN_DIR_SECONDS = 0
    msg.info(
        f"Sleeping for {CLEAN_DIR_SECONDS} seconds to allow watcher to start and clean dir",
    )
    time.sleep(CLEAN_DIR_SECONDS)

    train_models_for_each_cell_in_grid(
        cfg=cfg,
        possible_look_distances=possible_look_distances,
        config_file_name=CONFIG_FILE_NAME,
    )

    msg.good(
        f"Training finished. Stopping the watcher in {cfg.project.watcher.keep_alive_after_training_minutes} minutes...",
    )

    time.sleep(60 * cfg.project.watcher.keep_alive_after_training_minutes)
    watcher.kill()
    msg.good("Watcher stopped.")
