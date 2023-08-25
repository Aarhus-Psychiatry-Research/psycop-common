"""Utility functions for column name inference."""
import re
from collections.abc import Iterable
from typing import Union

import pandas as pd
from omegaconf import DictConfig

from psycop.common.model_training.config_schemas.full_config import FullConfigSchema


def get_col_names(
    cfg: Union[DictConfig, FullConfigSchema],
    dataset: pd.DataFrame,
) -> tuple[Union[str, list[str]], list[str]]:
    """Get column names for outcome and features.

    Args:
        cfg (DictConfig): Config object
        dataset: Dataset to get column names from

    Returns:
        outcome_col_name: Name of the outcome column
        train_col_names: Names of the columns to use for training
    """
    potential_outcome_col_names = [
        c
        for c in dataset.columns
        if cfg.data.outc_prefix in c
        and str(cfg.preprocessing.pre_split.min_lookahead_days) in c
    ]

    if cfg.preprocessing.pre_split.keep_only_one_outcome_col:
        if len(potential_outcome_col_names) != 1:
            raise ValueError(
                "More than one outcome column found. Please make outcome column names unambiguous.",
            )

        outcome_col_name = potential_outcome_col_names[0]

    else:
        outcome_col_name = potential_outcome_col_names

    train_col_names = [
        c
        for c in dataset.columns
        if c.startswith(cfg.data.pred_prefix) and "uuid" not in c
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

    if not isinstance(col_name, str):
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


def infer_col_names(
    df: pd.DataFrame,
    prefix: str,
    allow_multiple: bool = True,
) -> list[str]:
    """Infer col names based on prefix."""
    col_name = [c for c in df.columns if c.startswith(prefix)]

    if len(col_name) == 1:
        return col_name
    if len(col_name) > 1:
        if allow_multiple:
            return col_name
        raise ValueError(
            f"Multiple columns found and allow_multiple is {allow_multiple}.",
        )
    if not col_name:
        raise ValueError("No outcome col name inferred")
    raise ValueError("No outcomes inferred")


def infer_outcome_col_name(
    df: pd.DataFrame,
    prefix: str = "outc_",
    allow_multiple: bool = True,
) -> list[str]:
    """Infer the outcome column name from the dataframe."""
    return infer_col_names(df=df, prefix=prefix, allow_multiple=allow_multiple)


def infer_predictor_col_name(
    df: pd.DataFrame,
    prefix: str = "pred_",
    allow_multiple: bool = True,
) -> list[str]:
    """Get the predictors that are used in the model."""
    return infer_col_names(df=df, prefix=prefix, allow_multiple=allow_multiple)


def infer_y_hat_prob_col_name(
    df: pd.DataFrame,
    prefix: str = "y_hat_prob",
    allow_multiple: bool = False,
) -> list[str]:
    """Infer the y_hat_prob column name from the dataframe."""
    return infer_col_names(df=df, prefix=prefix, allow_multiple=allow_multiple)
