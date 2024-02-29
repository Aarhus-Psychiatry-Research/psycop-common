
from psycop.common.types.validated_frame import ValidatedFrame
import polars as pl

from dataclasses import dataclass

from psycop.common.types.validator_rules import ColumnExistsRule, ColumnTypeRule

@dataclass(frozen=True)
class MinimalEvalDataset(ValidatedFrame[pl.DataFrame]):
    pred_time_uuid_col_name: str
    pred_proba_col_name: str
    y_col_name: str

    pred_proba_col_rules = (
        ColumnExistsRule(),
        ColumnTypeRule(expected_type=pl.Int64)
    )