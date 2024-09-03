from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl
from iterpy import Iter
from polars import LazyFrame

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.preprocessing.step import PresplitStep


@dataclass(frozen=True)
class MissingColumnError(Exception):
    column_name: str


@BaselineRegistry.preprocessing.register("column_exists_validator")
@dataclass(frozen=True)
class ColumnExistsValidator(PresplitStep):
    column_names: Sequence[str]

    def apply(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        df = input_df.fetch(1) if isinstance(input_df, LazyFrame) else input_df  # type: ignore

        errors: list[MissingColumnError] = [
            self._column_name_exists(column_name=column, df=df)
            for column in self.column_names
            if self._column_name_exists(column_name=column, df=df) is not None
        ]  # type: ignore

        if errors:
            missing_columns_str = ", ".join([e.column_name for e in errors])
            raise MissingColumnError(f"Column(s) [{missing_columns_str}] not found in dataset.")

        return input_df

    def _column_name_exists(self, column_name: str, df: pl.DataFrame) -> MissingColumnError | None:
        if column_name not in df.columns:
            return MissingColumnError(column_name=column_name)

        return None


class ColumnCountError(Exception):
    ...


@dataclass(frozen=True)
class ColumnPrefixCountExpectation:
    prefix: str
    count: int

    @staticmethod
    def from_list(args: tuple[str, int]) -> "ColumnPrefixCountExpectation":
        if not len(args) == 2:
            raise ValueError(
                f"ColumnCountExpectation.from_list() takes exactly 2 arguments, ({len(args)} given)"
            )

        prefix = args[0]
        count = args[1]
        return ColumnPrefixCountExpectation(prefix=prefix, count=count)


@BaselineRegistry.preprocessing.register("column_prefix_count_expectation")
class ColumnPrefixExpectation(PresplitStep):
    def __init__(self, column_expectations: Sequence[Sequence[str | int]]):
        self.column_expectations = (
            Iter(column_expectations)
            .map(lambda x: ColumnPrefixCountExpectation.from_list(x))
            .to_list()
        )

    def apply(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        errors = (
            Iter(self.column_expectations)
            .map(
                lambda expectation: self._column_count_as_expected(
                    expectation=expectation, columns=input_df.columns
                )
            )
            .flatten()
            .to_list()
        )

        if errors:
            raise ColumnCountError(
                "Column count expectation(s) not met:\n\t" + "\n\t".join([str(e) for e in errors])
            )

        return input_df

    def _column_count_as_expected(
        self, expectation: ColumnPrefixCountExpectation, columns: Sequence[str]
    ) -> list[ColumnCountError]:
        matching_columns = [column for column in columns if column.startswith(expectation.prefix)]

        if len(matching_columns) != int(expectation.count):
            return [
                ColumnCountError(
                    f'"{expectation.prefix}" matched {matching_columns if matching_columns else "None"}, expected {expectation.count} matches.'
                )
            ]

        return []
