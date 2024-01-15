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
            "datetime_col": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        },
    )

    transformed = BoolToInt().apply(pl.from_pandas(df).lazy()).collect()
    assert transformed.get_column("bool_col").to_list() == [1, 0, 1]
    assert transformed["int_col"].dtype == pl.Int64
    assert transformed["str_col"].dtype == pl.Utf8
    assert transformed["datetime_col"].dtype == pl.Datetime
