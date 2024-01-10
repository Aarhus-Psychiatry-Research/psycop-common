"""Loaders for patient IDs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import polars as pl

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.common.model_training_v2.trainer.data.data_filters.geographical_split.make_geographical_split import (
    get_regional_split_df,
)

from ....types.validated_frame import ValidatedFrame
from ....types.validator_rules import ColumnExistsRule, ColumnTypeRule, ValidatorRule

if TYPE_CHECKING:
    from collections.abc import Sequence


class SplitName(Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


@dataclass(frozen=True)
class SplitFrame(ValidatedFrame[pl.LazyFrame]):
    split_name: str
    id_col_name: str = "dw_ek_borger"
    id_col_rules: Sequence[ValidatorRule] = (
        ColumnExistsRule(),
        ColumnTypeRule(expected_type=pl.Utf8),
    )


def load_stratified_by_outcome_split_ids(
    split: SplitName,
    n_rows: int | None = None,
) -> SplitFrame:
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

    return SplitFrame(
        frame=pl.from_pandas(df.reset_index(drop=True)).lazy(),
        split_name=split.value,
        allow_extra_columns=True,
    )


def load_stratified_by_region_split_ids(
    split: SplitName,
    n_rows: int | None = None,
) -> SplitFrame:
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

    return SplitFrame(
        frame=split_df.lazy(),
        split_name=split.value,
        allow_extra_columns=True,
    )
