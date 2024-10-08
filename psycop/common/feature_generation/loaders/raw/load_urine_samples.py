"""Loaders for medications."""

from __future__ import annotations

import logging

import pandas as pd

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load

log = logging.getLogger(__name__)


def urine_loader(n_rows: int | None = None) -> pd.DataFrame:
    """Load urine sample data.

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pd.DataFrame
    """

    sql = "SELECT * FROM [fct].[FOR_Mikrobiologi_urin_inkl_2021_okt2024]"

    df = sql_load(sql, database="USR_PS_FORSK", n_rows=n_rows)

    return df


def _pathogen_group_a():
    pass


def uvi_positive():
    pass


if __name__ == "__main__":
    urine_loader()
