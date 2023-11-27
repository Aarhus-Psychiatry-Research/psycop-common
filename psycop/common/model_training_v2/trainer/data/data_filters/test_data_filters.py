import polars as pl
import pytest
from psycop.common.model_training_v2.trainer.data.data_filters.geography import (
    subset_by_timestamp,
)


def test_time_subset():
    df1 = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 3],
            "timestamp": [
                "2022-01-01",
                "2022-02-01",
                "2022-03-01",
                "2022-03-01",
                "2022-03-01",
                "2022-03-01",
            ],
        }
    )

    df2 = pl.DataFrame(
        {"id": [1, 2, 3], "cutoff_timestamp": ["2022-02-15", None, None]}
    )
    df1 = df1.with_columns(pl.col("timestamp").str.strptime(pl.Date))
    df2 = df2.with_columns(pl.col("cutoff_timestamp").str.strptime(pl.Date))

    filtered_df = subset_by_timestamp(
        df1, df2, id_col_name="id", timestamp_col_name="timestamp"
    )

    assert filtered_df.shape[0] == 5
