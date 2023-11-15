import pytest
from polars import ColumnNotFoundError

from psycop.common.model_training_v2.loggers.base_logger import TerminalLogger
from psycop.common.model_training_v2.trainer.preprocessing.steps.col_filters import (
    LookbehindCombinationColFilter,
)
from psycop.common.model_training_v2.trainer.preprocessing.steps.column_validator import (
    ColumnExistsValidator,
)
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filters import (
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


def test_lookbehind_combination_filter():
    df = str_to_pl_df(
        """pred_age,pred_age_within_2_days,pred_age_within_3_days,pred_diagnosis_within_4_days
        3,4,3,2
        3,4,3,3
        4,3,4,1
        """,
    )

    logger = TerminalLogger()
    result = LookbehindCombinationColFilter(
        lookbehinds={2, 3},
        pred_col_prefix="pred_",
        logger=logger,
    ).apply(df)

    assert len(result.columns) == 3

def test_column_validator():
    df = str_to_pl_df("""
        pred_age,
        1,
    """,
    )

    # Check passing test
    ColumnExistsValidator("pred_age").apply(df)

    # Fail
    with pytest.raises(ValueError, match=r".+\[unknown_column\] not found in dataset.*"):
        ColumnExistsValidator("pred_age", "unknown_column").apply(df)
