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


@dataclass(frozen=True)
class FakeColnameValidatedFrame(ValidatedFrame[pl.DataFrame]):
    frame: pl.DataFrame
    test_col_name: str = "col_name"


def test_col_name_validation():
    df = str_to_pl_df(
        """test_col_name,
                      1,
""",
    )

    with pytest.raises(CombinedFrameValidationError, match=".*missing.*"):
        FakeColnameValidatedFrame(frame=df)


def test_type_validation():
    df = str_to_pl_df(
        """test_col_name,
                      1,
""",
    )

    @dataclass(frozen=True)
    class FakeTypeValidatedFrame(ValidatedFrame[pl.DataFrame]):
        frame: pl.DataFrame
        test_col_name: str = "test_col_name"
        test_col_rules: Sequence[ValidatorRule] = [ColumnTypeRule(pl.Int64)]

    with pytest.raises(CombinedFrameValidationError, match=".*type.*"):
        FakeTypeValidatedFrame(frame=df)


def test_rules_without_columns_error():
    @dataclass(frozen=True)
    class FakeFrameWithRulesWithoutColumns(ValidatedFrame[pl.DataFrame]):
        frame: pl.DataFrame
        test_col_rules: Sequence[ValidatorRule] = (ColumnTypeRule(pl.Int64),)

    with pytest.raises(
        CombinedFrameValidationError,
        match=".*missing from the frame.*",
    ):
        FakeFrameWithRulesWithoutColumns(frame=pl.DataFrame())
