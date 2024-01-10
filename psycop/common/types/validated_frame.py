from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from iterpy import Iter

from psycop.common.types.validator_rules import (
    ColumnExistsRule,
    ColumnInfo,
    ColumnMissingError,
    FrameValidationError,
    ValidatorRule,
)

from .polarsframe import PolarsFrameGeneric

T = TypeVar("T")


@dataclass(frozen=True)
class CombinedFrameValidationError(BaseException):
    error: str


@dataclass(frozen=True)
class ColumnAttrMissingError(FrameValidationError):
    attr: str

    def get_error_string(self) -> str:
        return f"- Attribute '{self.attr}' was expected in the dataclass, but was not present."


@dataclass(frozen=True)
class ColumnMissingInFrameError(FrameValidationError):
    col_name: str

    def get_error_string(self) -> str:
        return f"- Rules specified for '{self.col_name}', but it is missing from the frame."


@dataclass(frozen=True)
class ExtraColumnError(FrameValidationError):
    name: str

    def get_error_string(self) -> str:
        return f"- Column '{self.name}' was not specified in the dataclass, but was present in the frame. If extra columns are expected, set 'allow_extra_columns' on the dataclass to True."


@dataclass(frozen=True)
class ValidatedFrame(Generic[PolarsFrameGeneric]):
    """Validates a PolarsFrameGeneric based on the rules specified in the dataclass.

    It dynamically applies rules based on the attributes, where any attribute ending in:
    * '_col_name' is checked to be a column in the frame
    * '_col_rules' consists of rules for that column, and are checked. E.g. 'test_col_rules' will be applied to the column 'test'.
    """

    frame: PolarsFrameGeneric
    allow_extra_columns: bool = False

    def _try_get_attr(self, attr: str) -> Any | ColumnAttrMissingError:
        try:
            return getattr(self, attr)
        except AttributeError:
            return ColumnAttrMissingError(attr=attr)

    def _get_single_column_information(self, col_name_attr: str) -> ColumnInfo:
        try:
            rules: Sequence[ValidatorRule] = [
                *getattr(self, col_name_attr.replace("_col_name", "_col_rules")),
                ColumnExistsRule(),
            ]
        except AttributeError:
            rules = [ColumnExistsRule()]

        return ColumnInfo(
            attr=col_name_attr,
            name=getattr(self, f"{col_name_attr}"),
            rules=rules,
        )

    def _get_column_infos(self) -> Iter[ColumnInfo]:
        return (
            Iter(vars(self))
            .filter(lambda attr: "_col_name" in attr)
            .map(lambda col_name: self._get_single_column_information(col_name))
        )

    def _get_missing_column_errors(self) -> Sequence[FrameValidationError]:
        rules_with_missing_columns = (
            Iter(vars(self))
            .filter(lambda attr: "_col_rules" in attr)
            .map(lambda rule_attr: rule_attr.replace("_rules", "_name"))
            .map(lambda name_attr: self._try_get_attr(attr=name_attr))
        )

        column_rules_without_attr = [
            try_attr
            for try_attr in rules_with_missing_columns
            if isinstance(try_attr, ColumnAttrMissingError)
        ]

        missing_in_frame: list[ColumnMissingError] = (
            rules_with_missing_columns.filter(
                lambda try_attr: not isinstance(try_attr, ColumnAttrMissingError),
            )
            .filter(lambda col_name: col_name not in self.frame.columns)
            .map(
                lambda col_name: ColumnMissingInFrameError(col_name=col_name),  # type: ignore
            )
            .to_list()
        )

        return [
            *column_rules_without_attr,
            *missing_in_frame,
        ]

    def _get_extra_column_errors(
        self,
        column_infos: Iter[ColumnInfo],
    ) -> Sequence[ExtraColumnError]:
        dataclass_column_names = {ci.name for ci in column_infos}
        errors = (
            Iter(self.frame.columns)
            .filter(lambda column_name: column_name not in dataclass_column_names)
            .map(lambda column_name: ExtraColumnError(name=column_name))
            .to_list()
        )
        return errors

    def _get_rule_errors(
        self,
        column_infos: Iter[ColumnInfo],
    ) -> Sequence[FrameValidationError]:
        return (
            Iter([c_info.check_rules(self.frame) for c_info in column_infos])
            .flatten()
            .to_list()
        )

    def __post_init__(self):
        column_infos = self._get_column_infos()
        extra_columns_error_strings = self._get_extra_column_errors(column_infos)
        rule_errors = self._get_rule_errors(column_infos)
        missing_column_error_strings = self._get_missing_column_errors()

        error_strings = (
            Iter(
                [
                    *extra_columns_error_strings,
                    *rule_errors,
                    *missing_column_error_strings,
                ],
            )
            .map(lambda error: error.get_error_string())
            .to_list()
        )

        if error_strings:
            validation_error_string = "\n".join(error_strings)
            raise CombinedFrameValidationError(
                f"Dataframe did not pass validation. Errors:\n{validation_error_string}",
            )
