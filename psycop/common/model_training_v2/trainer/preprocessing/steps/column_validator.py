from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl
from functionalpy import Seq
from polars import LazyFrame, count

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.preprocessing.step import (
    PolarsFrame_T0,
    PresplitStep,
)


@dataclass(frozen=True)
class MissingColumnError(Exception):
    column_name: str


@BaselineRegistry.preprocessing.register("column_exists_validator")
class ColumnExistsValidator(PresplitStep):
    def __init__(
        self,
        *args: str,
    ):
        self.column_names = args

    def apply(self, input_df: PolarsFrame_T0) -> PolarsFrame_T0:
        df = input_df.fetch(1) if isinstance(input_df, LazyFrame) else input_df

        errors: list[MissingColumnError] = [
            self._column_name_exists(column_name=column, df=df)
            for column in self.column_names
            if self._column_name_exists(column_name=column, df=df) is not None
        ]  # type: ignore

        if errors:
            missing_columns_str = ", ".join([e.column_name for e in errors])
            raise MissingColumnError(
                f"Column(s) [{missing_columns_str}] not found in dataset.",
            )

        return input_df

    def _column_name_exists(
        self,
        column_name: str,
        df: pl.DataFrame,
    ) -> MissingColumnError | None:
        if column_name not in df.columns:
            return MissingColumnError(column_name=column_name)

        return None




class ColumnCountError(Exception):
    ...


@dataclass(frozen=True)
class ColumnCountExpectation:
    prefix: str
    count: int

    @classmethod
    def from_list(cls: type["ColumnCountExpectation"], args: list[str | int]) -> "ColumnCountExpectation":
        if not len(args) == 2:
            raise ValueError(
                f"ColumnCountExpectation.from_list() takes exactly 2 arguments, ({len(args)} given)",
            )

        prefix = args[0]
        count = args[1]  # noqa: F811
        return cls(prefix=prefix, count=count) # type: ignore


@BaselineRegistry.preprocessing.register("column_exists_validator")
class ColumnPrefixExpectation(PresplitStep):
    def __init__(
        self,
        *args: list[str | int],
    ):
        self.column_expectations = (
            Seq(args)
            .map(
                lambda x: ColumnCountExpectation.from_list(x),
            )
            .to_list()
        )

    def apply(self, input_df: PolarsFrame_T0) -> PolarsFrame_T0:
        df = input_df.fetch(1) if isinstance(input_df, LazyFrame) else input_df

        errors = (
            Seq(self.column_expectations)
            .map(
                lambda expectation: self._column_count_as_expected(
                    expectation=expectation, df=df,
                ),
            )
            .flatten()
            .to_iter()
        )

        if errors:
            raise ColumnCountError(
                "Column count expectation(s) not met:\n\t"
                + "\n\t".join([str(e) for e in errors]),
            )

        return input_df

    @staticmethod
    def _column_count_as_expected(
        expectation: ColumnCountExpectation,
        df: pl.DataFrame,
    ) -> list[ColumnCountError]:
        matching_columns = [
            column for column in df.columns if column.startswith(expectation.prefix)
        ]

        if len(matching_columns) != expectation.count:
            matched_columns_str = "\n\t".join(matching_columns)
            return [
                ColumnCountError(
                    f'Number of columns with prefix {expectation.prefix} does not match expectation of {expectation.count}. Columns that matched: {matched_columns_str if matching_columns else "None"}',
                ),
            ]

        return []
