from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import polars as pl
from iterpy import Iter

from .polarsframe import PolarsFrame


class FrameValidationError(Protocol):
    def get_error_string(self) -> str:
        ...


@dataclass(frozen=True)
class ColumnInfo:
    attr: str
    name: str
    rules: Sequence["ValidatorRule"]

    @property
    def specification_string(self) -> str:
        return f"Attr '{self.attr}' specifies '{self.name}'"

    def check_rules(self, frame: PolarsFrame) -> Sequence["FrameValidationError"]:
        return (
            Iter([rule(self, frame) for rule in self.rules if rule(self, frame)])
            .flatten()
            .to_list()
        )


class ValidatorRule(Protocol):
    def __call__(
        self, column_info: ColumnInfo, frame: PolarsFrame
    ) -> Sequence[FrameValidationError]:
        ...


@dataclass(frozen=True)
class ColumnMissingError(FrameValidationError):
    column: ColumnInfo

    def get_error_string(self) -> str:
        return f"{self.column.specification_string}. Column is missing from frame."


class ColumnExistsRule(ValidatorRule):
    def __call__(self, column_info: ColumnInfo, frame: PolarsFrame) -> Sequence[ColumnMissingError]:
        if column_info.name not in frame.columns:
            return [ColumnMissingError(column=column_info)]

        return []


@dataclass(frozen=True)
class ColumnTypeError(FrameValidationError):
    column: ColumnInfo
    actual_type: pl.PolarsDataType
    expected_type: pl.PolarsDataType

    def get_error_string(self) -> str:
        return f"{self.column.specification_string}. Expected type '{self.expected_type}', got '{self.actual_type}'."


@dataclass(frozen=True)
class ColumnTypeRule(ValidatorRule):
    expected_type: pl.PolarsDataType

    def __call__(self, column_info: ColumnInfo, frame: PolarsFrame) -> Sequence[ColumnTypeError]:
        try:
            column_type = frame.schema[column_info.name]
        except pl.ColumnNotFoundError:
            return []

        if column_type == self.expected_type:
            return []

        return [
            ColumnTypeError(
                column=column_info, actual_type=column_type, expected_type=self.expected_type
            )
        ]
