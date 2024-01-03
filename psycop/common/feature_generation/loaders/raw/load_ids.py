"""Loaders for patient IDs."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Literal

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.common.model_training_v2.trainer.data.data_filters.geographical_split.make_geographical_split import (
    get_regional_split_df,
)

if TYPE_CHECKING:
    import pandas as pd

import polars as pl


class SplitName(Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


def load_original_ids(split: SplitName, n_rows: int | None = None) -> pd.DataFrame:
    """Loads ids for a given split based on the original data split.

    Args:
        split: Which split to load IDs from. Takes either "train", "test" or "val".
        n_rows: Number of rows to return. Defaults to None.

    Returns:
        pd.DataFrame: Only dw_ek_borger column with ids
    """
    view = f"[psycop_{split.value}_ids]"

    sql = f"SELECT * FROM [fct].{view}"

    df = sql_load(sql, database="USR_PS_FORSK", n_rows=n_rows)

    return df.reset_index(drop=True)


def load_region_ids(split: SplitName, n_rows: int | None = None) -> pd.DataFrame:
    """Loads ids for a given split using the region-based split.

    Args:
        split: Which split to load IDs from. Takes either "train", "test" or "val".
        n_rows: Number of rows to return. Defaults to None.

    Returns:
        pd.DataFrame: Only dw_ek_borger column with ids
    """
    split_df = (
        get_regional_split_df()
        .filter(pl.col("split") == split.value)
        .select("dw_ek_borger")
        .collect()
    )
    if n_rows is not None:
        split_df = split_df.head(n_rows)
    return split_df.to_pandas()


def load_ids(
    split_type: Literal["original", "geographical"],
    split_name: SplitName,
    n_rows: int | None = None,
) -> pd.DataFrame:
    """Loads ids for a given split.

    Args:
        split_type: Which split to load IDs from. Takes either "original" or "geographical".
        split: Which split to load IDs from. Takes either "train", "test" or "val".
        n_rows: Number of rows to return. Defaults to None.

    Returns:
        pd.DataFrame: Only dw_ek_borger column with ids
    """
    if split_type == "original":
        return load_original_ids(split=split_name, n_rows=n_rows)
    elif split_type == "geographical":
        return load_region_ids(split=split_name, n_rows=n_rows)
    else:
        raise ValueError(f"split_type {split_type} is not allowed")
