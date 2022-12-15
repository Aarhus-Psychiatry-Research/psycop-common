"""Utility functions for column name inference."""
import pandas as pd
from omegaconf import DictConfig


def get_col_names(cfg: DictConfig, train: pd.DataFrame) -> tuple[str, list[str]]:
    """Get column names for outcome and features.

    Args:
        cfg (DictConfig): Config object
        train: Training dataset

    Returns:
        outcome_col_name: Name of the outcome column
        train_col_names: Names of the columns to use for training
    """

    potential_outcome_col_names = [
        c
        for c in train.columns
        if cfg.data.outc_prefix in c and str(cfg.data.min_lookahead_days) in c
    ]

    if len(potential_outcome_col_names) != 1:
        raise ValueError(
            "More than one outcome column found. Please make outcome column names unambiguous.",
        )

    outcome_col_name = potential_outcome_col_names[0]

    train_col_names = [  # pylint: disable=invalid-name
        c for c in train.columns if c.startswith(cfg.data.pred_prefix)
    ]

    return outcome_col_name, train_col_names
