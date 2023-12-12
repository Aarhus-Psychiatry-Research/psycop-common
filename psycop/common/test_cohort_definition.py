from datetime import datetime

import polars as pl

from psycop.common.types.polarsframe import PolarsFrameGeneric

from .cohort_definition import PredictionTimeFilter, filter_prediction_times
from .test_utils.str_to_df import str_to_pl_df


def test_filter_prediction_times():
    prediction_times = str_to_pl_df(
        """
        entity_id,  timestamp,
        1,          2020-01-01,
        1,          2018-01-01, # Filtered because of timestamp
        """,
    ).lazy()

    min_timestamp = datetime.strptime("2019-01-01", "%Y-%m-%d")

    class MinTimestampFilter(PredictionTimeFilter):
        def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
            return df.filter(pl.col("timestamp") > min_timestamp)

    filtered = filter_prediction_times(
        prediction_times=prediction_times,
        get_counts=False,
        filtering_steps=[MinTimestampFilter()],
        entity_id_col_name="entity_id",
    )

    assert len(filtered.prediction_times) == 1
