from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from iterpy import Iter

from psycop.common.types.validator_rules import (
    ColumnExistsRule,
    ColumnInfo,
    ValidatorRule,
)

from .polarsframe import PolarsFrameGeneric

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T")


@dataclass(frozen=True)
class CombinedFrameValidationError(BaseException):
    error: str


@dataclass(frozen=True)
class ValidatedFrame(Generic[PolarsFrameGeneric]):
    """Validates a PolarsFrameGeneric based on the rules specified in the dataclass.

    It dynamically applies rules based on the attributes, where any attribute ending in:
    * '_col_name' is checked to be a column in the frame
    * '_col_rules' consists of rules for that column, and are checked. E.g. 'test_col_rules' will be applied to the column 'test'.
    """

    frame: PolarsFrameGeneric

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

    def __post_init__(self):
        validation_errors = (
            Iter(
                [c_info.check_rules(self.frame) for c_info in self._get_column_infos()],
            )
            .flatten()
            .map(lambda error: error.get_error_string())
            .to_list()
        )

        rules_without_columns = (
            Iter(vars(self))
            .filter(lambda attr: "_col_rules" in attr)
            .map(lambda rule_attr: rule_attr.replace("_rules", "_name"))
            .map(lambda name_attr: getattr(self, name_attr))
            .filter(lambda col_name: col_name not in self.frame.columns)
            .map(
                lambda col_name: f"- Rules specified for '{col_name}', but it is missing from the frame.",
            )
            .to_list()
        )

        if validation_errors or rules_without_columns:
            validation_error_string = "\n".join(
                [*validation_errors, *rules_without_columns],
            )
            raise CombinedFrameValidationError(
                f"Dataframe did not pass validation. Errors:\n{validation_error_string}",
            )
