from typing import Literal

import polars as pl
import pytest

from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_other import (
    AgeFilter,
    DateFilter,
    QuarantineFilter,
    WindowFilter,
)

from .....test_utils.str_to_df import str_to_pl_df
from ...base_dataloader import BaselineDataLoader


@pytest.mark.parametrize(
    ("min_age", "max_age", "n_remaining"), [(1, 3, 3), (2, 3, 2), (1, 2, 2), (4, 4, 0)]
)
def test_age_filter(min_age: int, max_age: int, n_remaining: int):
    df = str_to_pl_df(
        """age,
    1,
    2,
    3,
    """
    ).lazy()

    result = AgeFilter(min_age=min_age, max_age=max_age, age_col_name="age").apply(df).collect()

    assert len(result) == n_remaining


@pytest.mark.parametrize(
    ("n_days", "direction", "n_remaining"),
    [(1, "ahead", 2), (2, "ahead", 1), (1, "behind", 2), (4, "behind", 0)],
)
def test_window_filter(n_days: int, direction: Literal["ahead", "behind"], n_remaining: int):
    df = str_to_pl_df(
        """timestamp,
    2021-01-01,
    2021-01-02,
    2021-01-03,
    2021-01-04,
    """
    ).lazy()

    result = (
        WindowFilter(n_days=n_days, direction=direction, timestamp_col_name="timestamp")
        .apply(df)
        .collect()
    )

    assert len(result) == n_remaining


def test_filter_by_quarantine_period():
    """Test filtering by quarantine date.

    Should filter if the prediction times lie within quarantine_interval_days after the prediction time.

    E.g. for type 2 diabetes, patients are quarantined for 730 days after they return to the Central Denmark Region.
    If a patient has an event in that time, they were probably incident outside of the region.
    Therefore, their following prediction times should be filtered out.

    Note that this function only filters the prediction times within the quarantine period.
    Filtering after the outcome is done inside TimeseriesFlattener when the OutcomeSpec has incident = True.
    """

    quarantine_days = 730

    class TestDataLoader(BaselineDataLoader):
        def load(self) -> pl.LazyFrame:
            return str_to_pl_df(
                """entity_id,timestamp,
                1,2021-01-01 00:00:01,
                1,2022-01-01 00:00:01,
                """
            ).lazy()

    prediction_time_df = str_to_pl_df(
        """entity_id,timestamp,
        1,2020-12-01 00:00:01, # keep: before quarantine date
        1,2022-12-01 00:00:01, # drop: within quarantine days from the first quarantine date
        1,2026-02-01 00:00:01, # keep: outside quarantine days from the first quarantine date
        2,2023-02-01 00:00:01, # keep: no quarantine date for this id
        """
    ).lazy()

    expected_df = str_to_pl_df(
        """entity_id,timestamp,
        1,2020-12-01 00:00:01,
        1,2026-02-01 00:00:01,
        2,2023-02-01 00:00:01,
        """
    )

    result_df = (
        QuarantineFilter(
            entity_id_col_name="entity_id",
            quarantine_interval_days=quarantine_days,
            quarantine_timestamps_loader=TestDataLoader(),
            timestamp_col_name="timestamp",
        )
        .apply(prediction_time_df)
        .collect()
    )

    # Check that the result is as expected using pandas.testing.assert_frame_equal
    from polars.testing import assert_frame_equal

    assert_frame_equal(result_df, expected_df)


@pytest.mark.parametrize(
    ("direction", "threshold_date", "expected_dates"),
    [
        ("before", "2023-03-01", ["2023-01-01", "2023-02-15"]),
        ("after-inclusive", "2023-03-30", ["2023-03-30", "2023-04-10", "2023-05-20"]),
    ],
)
def test_date_filter(
    direction: Literal["before", "after-inclusive"], threshold_date: str, expected_dates: list[str]
):
    df = str_to_pl_df(
        """timestamp
    2023-01-01
    2023-02-15
    2023-03-30
    2023-04-10
    2023-05-20
    """
    ).lazy()

    date_filter = DateFilter(
        column_name="timestamp", threshold_date=threshold_date, direction=direction
    )
    filtered_df = date_filter.apply(df).collect()

    assert filtered_df["timestamp"].cast(pl.Utf8).to_list() == [
        f"{date} 00:00:00.000000" for date in expected_dates
    ]
