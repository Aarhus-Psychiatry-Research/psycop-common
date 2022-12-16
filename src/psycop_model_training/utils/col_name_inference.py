"""Utility functions for column name inference."""
import re
from collections.abc import Iterable
from typing import Union

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
        if cfg.data.outc_prefix in c
        and str(cfg.preprocessing.pre_split.min_lookahead_days) in c
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


def infer_look_distance(
    col_name: Union[Iterable[str], str],
    regex_pattern: str = r"within_(\d+)_days",
    allow_multiple: bool = True,
) -> list[str]:
    """Infer look distances from col names."""
    # E.g. "outc_within_1_days" = 1
    # E.g. "outc_within_2_days" = 2
    # E.g. "pred_within_3_days" = 3
    # E.g. "pred_within_3_days" = 3

    look_distances: list[str] = []

    if isinstance(col_name, Iterable) and not isinstance(col_name, str):
        for c_name in col_name:
            look_distances += infer_look_distance(
                col_name=c_name,
                regex_pattern=regex_pattern,
            )
    else:
        look_distances = re.findall(pattern=regex_pattern, string=col_name)

    if len(look_distances) > 1 and not allow_multiple:
        raise ValueError(
            f"Multiple col names provided and allow_multiple is {allow_multiple}.",
        )

    return look_distances
