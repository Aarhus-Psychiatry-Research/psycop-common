from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl
import pytest

from ..test_utils.str_to_df import str_to_pl_df
from .validated_frame import (
    CombinedFrameValidationError,
    ValidatedFrame,
)
from .validator_rules import ColumnTypeRule, ValidatorRule


def test_col_name_validation():
    @dataclass(frozen=True)
    class FakeColnameValidatedFrame(ValidatedFrame[pl.DataFrame]):
        frame: pl.DataFrame
        test_col_name: str = "col_name"
        allow_extra_columns: bool = True

    df = str_to_pl_df(
        """test_col_name,
                      1,
""",
    )

    with pytest.raises(CombinedFrameValidationError, match=".*missing.*"):
        FakeColnameValidatedFrame(frame=df)


def test_type_validation():
    df = str_to_pl_df(
        """test,
                      1,
""",
    )

    @dataclass(frozen=True)
    class FakeTypeValidatedFrame(ValidatedFrame[pl.DataFrame]):
        frame: pl.DataFrame
        test_col_name: str = "test"
        test_col_rules: Sequence[ValidatorRule] = (ColumnTypeRule(pl.Utf8),)
        allow_extra_columns: bool = True

    with pytest.raises(CombinedFrameValidationError, match=".*type.*"):
        FakeTypeValidatedFrame(frame=df)


def test_rules_without_column_error():
    @dataclass(frozen=True)
    class FakeFrameWithRulesWithoutColumns(ValidatedFrame[pl.DataFrame]):
        frame: pl.DataFrame
        test_col_rules: Sequence[ValidatorRule] = (ColumnTypeRule(pl.Int64),)
        allow_extra_columns: bool = True

    with pytest.raises(
        CombinedFrameValidationError,
        match=".*was not present.*",
    ):
        FakeFrameWithRulesWithoutColumns(frame=pl.DataFrame())


def test_rules_for_non_attr_col_name():
    @dataclass(frozen=True)
    class RulesWithSeparateColName(ValidatedFrame[pl.DataFrame]):
        frame: pl.DataFrame = pl.DataFrame({"non_attr_col_name": [1]})  # noqa: RUF009
        test_col_name: str = "non_attr_col_name"
        test_col_rules: Sequence[ValidatorRule] = (ColumnTypeRule(pl.Int64),)
        allow_extra_columns: bool = True

    # Should not raise an error, since column name is gotten from the value of the "test_col_name" attribute
    RulesWithSeparateColName()


def test_should_error_if_unspecified_column_and_not_allow_extra():
    with pytest.raises(CombinedFrameValidationError, match=".*was present.*"):
        ValidatedFrame(frame=pl.DataFrame({"col": [1]}), allow_extra_columns=False)
