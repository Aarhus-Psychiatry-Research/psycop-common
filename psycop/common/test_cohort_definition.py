from datetime import datetime as dt

import polars as pl

from .cohort_definition import PredictionTimeFilter, filter_prediction_times
from .test_utils.str_to_df import str_to_pl_df


def test_filter_prediction_times():
    prediction_times = str_to_pl_df(
        """
        dw_ek_borger,  timestamp,
        1,          2020-01-01,
        1,          2019-01-01, # Filtered because of timestamp in filter 1
        1,          2018-01-01, # Filtered because of timestamp in filter 2
        """,
    ).lazy()

    class RemoveYear(PredictionTimeFilter):
        def __init__(self, min_timestamp: dt):
            self.year_timestamp = min_timestamp

        def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
            """Remove all prediction times within the year of the timestamp"""
            return df.filter(
                pl.col("timestamp").dt.year() != self.year_timestamp.year,
            )

    filtered = filter_prediction_times(
        prediction_times=prediction_times,
        get_counts=False,
        filtering_steps=[
            RemoveYear(dt.strptime("2018", "%Y")),
            RemoveYear(dt.strptime("2019", "%Y")),
        ],
        entity_id_col_name="entity_id",
    )

    assert len(filtered.prediction_times.frame) == 1
