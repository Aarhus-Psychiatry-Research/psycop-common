"""Loader for moves between regions."""
from typing import Optional

import pandas as pd

from psycop_feature_generation.loaders.raw.sql_load import sql_load


def load_moves(n_rows: Optional[int] = None) -> pd.DataFrame:
    """Get a dataframe with timestamps for each move into or out from the
    Central Denmark Region."""
    view = "[bopael_i_rm]"

    sql = f"SELECT * FROM [fct].{view}"

    df = sql_load(sql, database="USR_PS_FORSK", chunksize=None, n_rows=n_rows)

    return df


def load_move_into_rm_for_exclusion(n_rows: Optional[int] = None) -> pd.DataFrame:
    """Get a dataframe with timestamps for each move into or out from the
    Central Denmark Region."""
    df = load_moves(n_rows=n_rows)

    df = df.drop(columns=["fraflytnings_datotid"])
    df = df.rename(columns={"tilflytnings_datotid": "timestamp"})
    df = df.loc[df["timestamp"] >= pd.to_datetime("2012-01-01")]

    return df


if __name__ == "__main__":
    df = load_move_into_rm_for_exclusion()
