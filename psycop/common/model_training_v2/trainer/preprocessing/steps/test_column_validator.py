import polars as pl
import pytest

from psycop.common.model_training_v2.trainer.preprocessing.steps.column_validator import (
    ColumnCountError,
    ColumnExistsValidator,
    ColumnPrefixExpectation,
    MissingColumnError,
)
from psycop.common.test_utils.str_to_df import str_to_pl_df


def test_columns_exist_validator():
    df = (
        str_to_pl_df(
            """
        pred_age,
        1,
    """,
        )
        .lazy()
        .filter(pl.col("pred_age") != pl.lit(1))
    )
    # We use .fetch() in ColumnExistsValidator, which can return a dataframe with no rows. Ensure the validator still works in that case.

    # Check passing test
    ColumnExistsValidator("pred_age").apply(df)

    # Fail
    with pytest.raises(
        MissingColumnError,
        match=r".+\[unknown_column\] not found in dataset.*",
    ):
        ColumnExistsValidator("pred_age", "unknown_column").apply(df)


def test_column_prefix_expectation():
    df = str_to_pl_df(
        """
        pred_age,
        1,
    """,
    ).lazy()

    # Check passing test
    ColumnPrefixExpectation(["pred_", 1]).apply(df)

    # Fail
    with pytest.raises(
        ColumnCountError,
        match=r".+count expectation.*",
    ):
        ColumnPrefixExpectation(["pred_", 2]).apply(df)
