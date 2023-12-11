from typing import Literal

import pytest

from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filters import (
    AgeFilter,
    WindowFilter,
)
from psycop.common.test_utils.str_to_df import str_to_pl_df


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
    )

    result = AgeFilter(min_age=min_age, max_age=max_age, age_col_name="age").apply(df)

    assert len(result) == n_remaining


@pytest.mark.parametrize(
    ("n_days", "direction", "timestamp_col_name", "n_remaining"),
    [
        (1, "ahead", "timestamp", 2),
        (2, "ahead", "timestamp", 1),
        (1, "behind", "timestamp", 2),
        (4, "behind", "timestamp", 0),
    ],
)
def test_window_filter(
    n_days: int,
    direction: Literal["ahead", "behind"],
    timestamp_col_name: str,
    n_remaining: int,
):
    df = str_to_pl_df(
        """timestamp,
    2021-01-01,
    2021-01-02,
    2021-01-03,
    2021-01-04,
    """,
    )

    result = WindowFilter(
        n_days=n_days,
        direction=direction,
        timestamp_col_name=timestamp_col_name,
    ).apply(df)

    assert len(result) == n_remaining
