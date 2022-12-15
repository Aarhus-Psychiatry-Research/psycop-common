"""Example script to train multiple models and subsequently log the results to
wandb.

Usage:
- Replace the HYDRA_ARGS string with the desired arguments for `train_model.py`
- Run this script from project root with `python src/psycopt2d/train_and_log_models.py
"""
import random
import subprocess
import time
from typing import Optional

import pandas as pd
import wandb
from psycopt2d.utils.config_schemas import (
    BaseModel,
    FullConfigSchema,
    load_cfg_as_pydantic,
)
from random_word import RandomWords
from wasabi import Printer

from psycop_model_training.config.schemas import FullConfigSchema, load_cfg_as_pydantic
from psycop_model_training.load import load_train_raw
from psycop_model_training.model_eval.evaluate_model import (
    infer_look_distance,
    infer_outcome_col_name,
)


def start_trainer(
    cfg: FullConfigSchema,
    config_file_name: str,
    lookahead_days: int,
    wandb_group_override: str,
    model_name: str,
) -> subprocess.Popen:
    """Start a trainer."""
    msg = Printer(timestamp=True)

    subprocess_args: list[str] = [
        "python",
        "src/psycopt2d/train_model.py",
        f"project.wandb.group='{wandb_group_override}'",
        f"project.wandb.mode={cfg.project.wandb.mode}",
        f"hydra.sweeper.n_trials={cfg.train.n_trials_per_lookahead}",
        f"hydra.sweeper.n_jobs={cfg.train.n_jobs_per_trainer}",
        f"model={model_name}",
        f"model={cfg.model.name}",
        f"data.min_lookahead_days={lookahead_days}",
        "--config-name",
        f"{config_file_name}",
    ]

    if cfg.train.n_trials_per_lookahead > 1:
        subprocess_args.insert(2, "--multirun")

    if model.name == "xgboost":
        subprocess_args.insert(3, "++model.args.tree_method='gpu_hist'")

    msg.info(f'{" ".join(subprocess_args)}')

    return subprocess.Popen(  # pylint: disable=consider-using-with
        args=subprocess_args,
    )


class TrainerSpec(BaseModel):
    """Specification for starting a trainer.
    Provides overrides for the config file.
    """

    lookahead_days: int
    model_name: str


def combine_lookaheads_and_model_names_to_trainer_specs(
    cfg: FullConfigSchema,
    possible_lookahead_days: list[int],
    model_names: Optional[list[str]] = None,
):
    """Generate trainer specs for all combinations of lookaheads and model
    names."""
    msg = Printer(timestamp=True)

    random.shuffle(possible_lookahead_days)

    if model_names:
        msg.warn(
            "model_names was specified in train_models_for_each_cell_in_grid, overriding cfg.model.name",
        )

    model_name_queue = model_names if model_names else cfg.model.name

    # Create all combinations of lookahead_days and models
    trainer_combinations_queue = [
        TrainerSpec(lookahead_days=lookahead_days, model_name=model_name)
        for lookahead_days in possible_lookahead_days.copy()
        for model_name in model_name_queue
    ]

    return trainer_combinations_queue


def train_models_for_each_cell_in_grid(
    cfg: FullConfigSchema,
    possible_lookahead_days: list[int],
    config_file_name: str,
    wandb_prefix: str,
    model_names: Optional[list[str]] = None,
):
    """Train a model for each cell in the grid of possible look distances."""
    active_trainers: list[subprocess.Popen] = []

    trainer_combinations_queue = combine_lookaheads_and_model_names_to_trainer_specs(
        cfg=cfg,
        possible_lookahead_days=possible_lookahead_days,
        model_names=model_names,
    )

    while trainer_combinations_queue or active_trainers:
        # Wait until there is a free slot in the trainers group
        if (
            len(active_trainers) >= cfg.train.n_active_trainers
            or len(trainer_combinations_queue) == 0
        ):
            # Drop trainers if they have finished
            # If finished, t.poll() is not None
            active_trainers = [t for t in active_trainers if t.poll() is None]
            time.sleep(1)
            continue

        # Start a new trainer
        trainer_spec = trainer_combinations_queue.pop()

        msg = Printer(timestamp=True)
        msg.info(
            f"Spawning a new trainer with lookahead={trainer_spec.lookahead_days} days",
        )
        wandb_group = f"{wandb_prefix}"

        active_trainers.append(
            start_trainer(
                cfg=cfg,
                config_file_name=config_file_name,
                lookahead_days=trainer_spec.lookahead_days,
                wandb_group_override=wandb_group,
                model_name=trainer_spec.model_name,
            ),
        )


def get_possible_lookaheads(
    msg: Printer,
    cfg: FullConfigSchema,
    train_df: pd.DataFrame,
) -> list[int]:
    """Some look_ahead and look_behind distances will result in 0 valid
    prediction times. Only return combinations which will allow some prediction
    times.

    E.g. if we only have 4 years of data:
    - min_lookahead = 2 years
    - min_lookbehind = 3 years

    Will mean that no rows satisfy the criteria.
    """

    outcome_col_names = infer_outcome_col_name(df=train_df, allow_multiple=True)

    possible_lookahead_days: list[int] = [
        int(dist) for dist in infer_look_distance(col_name=outcome_col_names)
    ]

    # Don't try look distance combinations which will result in 0 rows
    max_distance_in_dataset_days = (
        max(train_df[cfg.data.col_name.pred_timestamp])
        - min(
            train_df[cfg.data.col_name.pred_timestamp],
        )
    ).days

    lookaheads_without_rows: list[int] = [
        dist for dist in possible_lookahead_days if dist > max_distance_in_dataset_days
    ]

    if lookaheads_without_rows:
        msg.info(
            f"Not fitting model to {lookaheads_without_rows}, since no rows satisfy the criteria.",
        )

    return list(set(possible_lookahead_days) - set(lookaheads_without_rows))


def main():
    """Main."""
    msg = Printer(timestamp=True)

    debug = False

    if debug:
        config_file_name = "integration_config.yaml"
    else:
        config_file_name = "default_config.yaml"

    cfg = load_cfg_as_pydantic(config_file_name=config_file_name)

    random_word = RandomWords()
    wandb_group = f"{random_word.get_random_word()}-{random_word.get_random_word()}"

    wandb.init(
        project=cfg.project.name,
        mode=cfg.project.wandb.mode,
        group=wandb_group,
        entity=cfg.project.wandb.entity,
        name="process_manager",
    )

    # Load dataset without dropping any rows for inferring
    # which look distances to grid search over
    train = load_train_raw(cfg=cfg)

    possible_lookaheads = get_possible_lookaheads(
        msg=msg,
        cfg=cfg,
        train_df=train,
    )

    train_models_for_each_cell_in_grid(
        cfg=cfg,
        possible_lookahead_days=possible_lookaheads,
        config_file_name=config_file_name,
        wandb_prefix=wandb_group,
        model_names=["xgboost", "logistic-regression"],
    )


if __name__ == "__main__":
    main()
