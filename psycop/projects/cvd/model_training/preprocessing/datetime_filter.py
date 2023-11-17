from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl
import polars.selectors as cs
from functionalpy import Seq
from polars import LazyFrame

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.preprocessing.step import (
    PolarsFrame_T0,
    PresplitStep,
)


@BaselineRegistry.preprocessing.register("temporal_col_filter")
class TemporalColumnFilter(PresplitStep):
    def __init__(self):
        pass

    def apply(self, input_df: PolarsFrame_T0) -> PolarsFrame_T0:
        temporal_columns = input_df.select(cs.temporal()).columns
        return input_df.drop(temporal_columns)
