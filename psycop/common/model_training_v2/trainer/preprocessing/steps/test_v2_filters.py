import pytest

from psycop.common.model_training_v2.trainer.preprocessing.steps.filters import (
    AgeFilter,
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
