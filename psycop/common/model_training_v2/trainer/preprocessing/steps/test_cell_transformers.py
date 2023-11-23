import pandas as pd
import polars as pl

from .cell_transformers import BoolToInt


def test_bool_to_int():
    # Use pandas df because that is the typical input
    df = pd.DataFrame(
        {
            "bool_col": [True, False, True],
            "int_col": [1, 2, 3],
            "str_col": ["a", "b", "c"],
        },
    )

    transformed = BoolToInt().apply(pl.from_pandas(df))
    assert transformed.get_column("bool_col").to_list() == [1, 0, 1]
    assert transformed["int_col"].dtype == pl.Int64
    assert transformed["str_col"].dtype == pl.Utf8
