from typing import Optional

import pandas as pd

from psycop_feature_generation.loaders.raw.sql_load import sql_load


def load_moves(n_rows: Optional[int] = None) -> pd.DataFrame:
    """Get a dataframe with timestamps for each move."""
    view = f"[bopael_i_rm]"

    sql = f"SELECT * FROM [fct].{view}"

    df = sql_load(sql, database="USR_PS_FORSK", chunksize=None, n_rows=n_rows)

    return df.reset_index(drop=True)
