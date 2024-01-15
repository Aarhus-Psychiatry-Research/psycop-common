from typing import Literal

import polars as pl
import pytest

from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_other import (
    AgeFilter,
    QuarantineFilter,
    WindowFilter,
)

from .....test_utils.str_to_df import str_to_pl_df
from ...base_dataloader import BaselineDataLoader


@pytest.mark.parametrize(
    ("min_age", "max_age", "n_remaining"),
    [
        (1, 3, 3),
        (2, 3, 2),
        (1, 2, 2),
        (4, 4, 0),
    ],
)
def test_age_filter(min_age: int, max_age: int, n_remaining: int):
    df = str_to_pl_df(
        """age,
    1,
    2,
    3,
    """,
    ).lazy()

    result = (
        AgeFilter(min_age=min_age, max_age=max_age, age_col_name="age")
        .apply(df)
        .collect()
    )

    assert len(result) == n_remaining


@pytest.mark.parametrize(
    ("n_days", "direction", "n_remaining"),
    [
        (1, "ahead", 2),
        (2, "ahead", 1),
        (1, "behind", 2),
        (4, "behind", 0),
    ],
)
def test_window_filter(
    n_days: int,
    direction: Literal["ahead", "behind"],
    n_remaining: int,
):
    df = str_to_pl_df(
        """timestamp,
    2021-01-01,
    2021-01-02,
    2021-01-03,
    2021-01-04,
    """,
    ).lazy()

    result = (
        WindowFilter(
            n_days=n_days,
            direction=direction,
            timestamp_col_name="timestamp",
        )
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

    class TestDataLoader(BaselineDataLoader):
        def load(self) -> pl.LazyFrame:
            return str_to_pl_df(
                """entity_id,timestamp,
                1,2021-01-01 00:00:01,
                1,2022-01-01 00:00:01,
                """,
            ).lazy()

    prediction_time_df = str_to_pl_df(
        """entity_id,timestamp,
        1,2020-12-01 00:00:01, # keep: before quarantine date
        1,2022-12-01 00:00:01, # drop: after quarantine date
        1,2026-02-01 00:00:01, # keep: outside quarantine days
        2,2023-02-01 00:00:01, # keep: no quarantine date for this id
        """,
        add_pred_time_uuid=True,
    ).lazy()

    expected_df = str_to_pl_df(
        """entity_id,timestamp,
        1,2020-12-01 00:00:01,
        1,2026-02-01 00:00:01,
        2,2023-02-01 00:00:01,
        """,
        add_pred_time_uuid=True,
    ).lazy()

    result_df = QuarantineFilter(
        entity_id_col_name="entity_id",
        quarantine_interval_days=730,
        quarantine_timestamps_loader=TestDataLoader(),
        timestamp_col_name="timestamp",
        pred_time_uuid_col_name="pred_time_uuid",
    ).apply(prediction_time_df)

    # Check that the result is as expected using pandas.testing.assert_frame_equal
    from polars.testing import assert_frame_equal

    assert_frame_equal(
        result_df,
        expected_df,
    )
