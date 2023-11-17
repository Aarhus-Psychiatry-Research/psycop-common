from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl
import polars.selectors as cs
from functionalpy import Seq
from polars import Boolean, LazyFrame

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.preprocessing.step import (
    PolarsFrame_T0,
    PresplitStep,
)


@BaselineRegistry.preprocessing.register("bool_to_int")
class BoolToInt(PresplitStep):
    def __init__(self):
        pass

    def apply(self, input_df: PolarsFrame_T0) -> PolarsFrame_T0:
        for col_name in input_df.columns:
            if input_df.schema[col_name] == Boolean: # type: ignore
                input_df = input_df.with_columns(pl.col(col_name).cast(int))

        return input_df
