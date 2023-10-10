"""Loaders for patient IDs."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load

if TYPE_CHECKING:
    import pandas as pd


class SplitName(Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


def load_ids(split: SplitName, n_rows: int | None = None) -> pd.DataFrame:
    """Loads ids for a given split.

    Args:
        split (str): Which split to load IDs from. Takes either "train", "test" or "val". # noqa: DAR102
        n_rows: Number of rows to return. Defaults to None.

    Returns:
        pd.DataFrame: Only dw_ek_borger column with ids
    """
    view = f"[psycop_{split.value}_ids]"

    sql = f"SELECT * FROM [fct].{view}"

    df = sql_load(sql, database="USR_PS_FORSK", n_rows=n_rows)

    return df.reset_index(drop=True)
