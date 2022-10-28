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
from typing import Union

import pandas as pd
from hydra import compose, initialize
from pydantic import BaseModel
from wasabi import Printer

from psycopt2d.evaluate_saved_model_predictions import (
    infer_look_distance,
    infer_outcome_col_name,
    infer_predictor_col_name,
)
from psycopt2d.load import DataLoader
from psycopt2d.utils.configs import FullConfigSchema, omegaconf_to_pydantic_objects

msg = Printer(timestamp=True)


class LookDistance(BaseModel):
    """A distance of ahead and behind."""

    behind_days: Union[int, float]
    ahead_days: Union[int, float]


def load_train_raw(cfg: FullConfigSchema):
    """Load the data."""
    path = Path(cfg.data.dir)
    file_names = list(path.glob(pattern=r"*train*"))

    if len(file_names) == 1:
        file_name = file_names[0]
        file_suffix = file_name.suffix
        if file_suffix == ".parquet":
            df = pd.read_parquet(file_name)
        elif file_suffix == ".csv":
            df = pd.read_csv(file_name)

        df = DataLoader.convert_timestamp_dtype_and_nat(dataset=df)

        return df

    raise ValueError(f"Returned {len(file_names)} files")


def infer_possible_look_distances(df: pd.DataFrame) -> list[LookDistance]:
    """Infer the possible values for min_lookahead_days and
    min_lookbehind_days."""
    # Get potential lookaheads from outc_ columns
    outcome_col_names = infer_outcome_col_name(df=df, allow_multiple=True)

    possible_lookahead_days = infer_look_distance(col_name=outcome_col_names)

    # Get potential lookbehinds from pred_ columns
    pred_col_names = infer_predictor_col_name(df=df, allow_multiple=True)
    possible_lookbehind_days = list(set(infer_look_distance(col_name=pred_col_names)))

    return [
        LookDistance(
            behind_days=lookbehind_days,
            ahead_days=lookahead_days,
        )
        for lookahead_days in possible_lookahead_days
        for lookbehind_days in possible_lookbehind_days
    ]


def start_trainer(
    cfg: FullConfigSchema,
    config_file_name: str,
    cell: LookDistance,
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


def train_models_for_each_cell_in_grid(
    cfg: FullConfigSchema,
    possible_look_distances: list[LookDistance],
    config_file_name: str,
):
    """Train a model for each cell in the grid of possible look distances."""
    from random_word import RandomWords

    random_word = RandomWords()

    random.shuffle(possible_look_distances)

    active_trainers: list[subprocess.Popen] = []

    wandb_prefix = f"{random_word.get_random_word()}-{random_word.get_random_word()}"

    while possible_look_distances or active_trainers:
        # Wait until there is a free slot in the trainers group
        if len(active_trainers) >= cfg.train.n_active_trainers:
            # Drop trainers if they have finished
            # If finished, t.poll() is not None
            active_trainers = [t for t in active_trainers if t.poll() is None]
            time.sleep(1)
            continue

        # Start a new trainer

        combination = possible_look_distances.pop()

        msg.info(
            f"Spawning a new trainer with lookbehind={combination.behind_days} and lookahead={combination.ahead_days}",
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


def load_cfg(config_file_name) -> FullConfigSchema:
    """Load config as pydantic object."""
    with initialize(version_base=None, config_path="../src/psycopt2d/config/"):
        cfg = compose(
            config_name=config_file_name,
        )

        cfg = omegaconf_to_pydantic_objects(cfg)
    return cfg


def get_possible_look_distances(
    msg: Printer,
    cfg: FullConfigSchema,
    train_df: pd.DataFrame,
) -> list[LookDistance]:
    """Some look_ahead and look_behind distances will result in 0 valid
    prediction times. Only return combinations which will allow some prediction
    times.

    E.g. if we only have 4 years of data:
    - min_lookahead = 2 years
    - min_lookbehind = 3 years

    Will mean that no rows satisfy the criteria.
    """

    look_combinations_in_dataset = infer_possible_look_distances(df=train_df)

    # Don't try look distance combinations which will result in 0 rows
    max_distance_in_dataset_days = (
        max(train_df[cfg.data.col_name.pred_timestamp])
        - min(
            train_df[cfg.data.col_name.pred_timestamp],
        )
    ).days

    look_combinations_without_rows = [
        dist
        for dist in look_combinations_in_dataset
        if (dist.ahead_days + dist.behind_days) > max_distance_in_dataset_days
    ]

    if look_combinations_without_rows:
        msg.info(
            f"Not fitting model to {look_combinations_without_rows}, since no rows satisfy the criteria.",
        )

    look_combinations_with_rows = [
        dist
        for dist in look_combinations_in_dataset
        if ((dist.ahead_days + dist.behind_days) < max_distance_in_dataset_days)
    ]

    return look_combinations_with_rows


def main():
    """Main."""
    msg = Printer(timestamp=True)

    config_file_name = "integration_config.yaml"

    cfg = load_cfg(config_file_name=config_file_name)

    # Load dataset without dropping any rows for inferring
    # which look distances to grid search over
    train = load_train_raw(cfg=cfg)

    possible_look_distances = get_possible_look_distances(
        msg=msg,
        cfg=cfg,
        train_df=train,
    )

    if not cfg.train.gpu:
        msg.warn("Not using GPU for training")

    train_models_for_each_cell_in_grid(
        cfg=cfg,
        possible_look_distances=possible_look_distances,
        config_file_name=config_file_name,
    )


if __name__ == "__main__":
    main()
