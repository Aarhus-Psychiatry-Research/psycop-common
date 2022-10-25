"""Example script to train multiple models and subsequently log the results to
wandb.

Usage:
- Replace the HYDRA_ARGS string with the desired arguments for `train_model.py`
- Run this script from project root with `python src/psycopt2d/train_and_log_models.py
"""
import random
import subprocess
import time
from pathlib import Path

import pandas as pd
from hydra import compose, initialize
from pydantic import BaseModel
from wasabi import Printer

from psycopt2d.evaluate_saved_model_predictions import (
    infer_look_distance,
    infer_outcome_col_name,
    infer_predictor_col_name,
)
from psycopt2d.load import load_train_from_cfg
from psycopt2d.utils.configs import FullConfig, omegaconf_to_pydantic_objects

msg = Printer(timestamp=True)


class PossibleLookDistanceDays(BaseModel):
    """Possible look distances."""

    ahead: list[str]
    behind: list[str]


def load_train_raw(cfg: FullConfig):
    """Load the data."""
    path = Path(cfg.data.dir)
    file = list(path.glob(pattern=r"*train*"))

    if len(file) == 1:
        return pd.read_parquet(file)

    raise ValueError(f"Returned {len(file)} files")


def infer_possible_look_distances(df: pd.DataFrame) -> PossibleLookDistanceDays:
    """Infer the possible values for min_lookahead_days and
    min_lookbehind_days."""
    # Get potential lookaheads from outc_ columns
    outcome_col_names = infer_outcome_col_name(df=df, allow_multiple=True)

    possible_lookahead_days = infer_look_distance(col_name=outcome_col_names)

    # Get potential lookbehinds from pred_ columns
    pred_col_names = infer_predictor_col_name(df=df, allow_multiple=True)
    possible_lookbehind_days = list(set(infer_look_distance(col_name=pred_col_names)))

    return PossibleLookDistanceDays(
        ahead=possible_lookahead_days,
        behind=possible_lookbehind_days,
    )


class LookDirectionCombination(BaseModel):
    """A combination of lookbehind and lookahead days."""

    behind_days: int
    ahead_days: int


def start_trainer(
    cfg: FullConfig,
    config_file_name: str,
    cell: LookDirectionCombination,
    wandb_group_override: str,
) -> subprocess.Popen:
    """Start a trainer."""
    subprocess_args: list[str] = [
        "python",
        "src/psycopt2d/train_model.py",
        f"model={cfg.model.model_name}",
        f"data.min_lookbehind_days={max(cfg.data.lookbehind_combination)}",
        f"data.min_lookahead_days={cell.ahead_days}",
        f"project.wandb.group='{wandb_group_override}'",
        f"hydra.sweeper.n_trials={cfg.train.n_trials_per_lookdirection_combination}",
        f"project.wandb.mode={cfg.project.wandb.mode}",
        "--config-name",
        f"{config_file_name}",
    ]

    if cfg.train.n_trials_per_lookdirection_combination > 1:
        subprocess_args.insert(2, "--multirun")

    if cfg.model.model_name == "xgboost" and not cfg.train.gpu:
        subprocess_args.insert(3, "++model.args.tree_method='auto'")

    msg.info(f'{" ".join(subprocess_args)}')

    return subprocess.Popen(  # pylint: disable=consider-using-with
        args=subprocess_args,
    )


def start_watcher(cfg: FullConfig) -> subprocess.Popen:
    """Start a watcher."""
    return subprocess.Popen(  # pylint: disable=consider-using-with
        [
            "python",
            "src/psycopt2d/model_training_watcher.py",
            "--entity",
            cfg.project.wandb.entity,
            "--project_name",
            cfg.project.name,
            "--n_runs_before_eval",
            str(cfg.project.watcher.n_runs_before_eval),
            "--overtaci",
            str(cfg.eval.save_model_predictions_on_overtaci),
            "--timeout",
            "None",
            "--clean_wandb_dir",
            str(cfg.project.watcher.archive_all),
            "--verbose",
            "True",
        ],
    )


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

    random.shuffle(lookbehind_combinations)

    active_trainers: list[subprocess.Popen] = []

    wandb_prefix = f"{random_word.get_random_word()}-{random_word.get_random_word()}"
    watcher = start_watcher(cfg=cfg)
    while lookbehind_combinations or active_trainers:
        # Wait until there is a free slot in the trainers group
        if len(active_trainers) >= cfg.train.n_active_trainers:
            # Drop trainers if they have finished
            # If finished, t.poll() is not None
            active_trainers = [t for t in active_trainers if t.poll() is None]
            time.sleep(1)
            continue

        # Start a new trainer

        combination = lookbehind_combinations.pop()

        # Check if any rows in the given combinatin of lookbehind and lookahead days
        cfg_for_checking_any_rows = cfg.copy()
        cfg_for_checking_any_rows.data.min_lookbehind_days = combination.lookbehind
        cfg_for_checking_any_rows.data.min_lookahead_days = combination.lookahead

        train = load_train_from_cfg(cfg=cfg)

        if train.shape[0] == 0:
            msg.warn(f"No rows for {combination}, continuing")
            continue

        # watcher = start_watcher(cfg=cfg)
        msg.info(
            f"Spawning a new trainer with lookbehind={combination.lookbehind} and lookahead={combination.lookahead}",
        )
        wandb_group = f"{wandb_prefix}"
        active_trainers.append(
            start_trainer(
                cfg=cfg,
                config_file_name=config_file_name,
                cell=combination,
                wandb_group_override=wandb_group,
            ),
        )

    msg.good(
        f"Training finished. Stopping the watcher in {cfg.project.watcher.keep_alive_after_training_minutes} minutes...",
    )

    time.sleep(60 * cfg.project.watcher.keep_alive_after_training_minutes)
    watcher.kill()


def load_cfg(config_file_name):
    """Load config as pydantic object."""
    with initialize(version_base=None, config_path="../src/psycopt2d/config/"):
        cfg = compose(
            config_name=config_file_name,
        )

        cfg = omegaconf_to_pydantic_objects(cfg)
    return cfg


def main():
    """Main."""
    msg = Printer(timestamp=True)

    config_file_name = "default_config.yaml"

    cfg = load_cfg(config_file_name=config_file_name)

    if cfg.project.wandb.mode == "run":
        msg.warn(
            f"wandb.mode is {cfg.project.wandb.mode}, not using the watcher. This will substantially slow down training.",
        )

    train = load_train_raw(cfg=cfg)
    possible_look_distances = get_possible_look_distances(msg, cfg, train)

    if not cfg.train.gpu:
        msg.warn("Not using GPU for training")

    train_models_for_each_cell_in_grid(
        cfg=cfg,
        possible_look_distances=possible_look_distances,
        config_file_name=config_file_name,
    )


def get_possible_look_distances(msg: Printer, cfg: FullConfig, train: pd.DataFrame):
    """Some look_ahead and look_behind distances will result in 0 valid
    prediction times. Only return combinations which will allow some prediction
    times.

    E.g. if we only have 4 years of data:
    - min_lookahead = 2 years
    - min_lookbehind = 3 years

    Will mean that no rows satisfy the criteria.
    """

    possible_look_distances = infer_possible_look_distances(df=train)

    lookbehind_combinations = [
        LookDirectionCombination(behind_days=behind_days, ahead_days=ahead_days)
        for behind_days in possible_look_distances.behind
        for ahead_days in possible_look_distances.ahead
    ]

    # Don't try look distance combinations which will result in 0 rows
    max_date_interval_in_dataset = max(train[cfg.data.pred_timestamp_col_name]) - max(
        train[cfg.data.pred_timestamp_col_name],
    )

    possible_look_distances = [
        dist
        for dist in lookbehind_combinations
        if ((dist.ahead + dist.behind_days) < max_date_interval_in_dataset)
    ]

    # Remove "9999" from possible look distances behind
    if cfg.data.max_lookbehind_days:
        possible_look_distances.behind = [
            dist
            for dist in possible_look_distances.behind
            if int(dist) <= cfg.data.max_lookbehind_days
        ]

    msg.info(f"Possible lookbehind days: {possible_look_distances.behind}")
    msg.info(f"Possible lookahead days: {possible_look_distances.ahead}")
    return possible_look_distances


if __name__ == "__main__":
    main()
