import polars as pl

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.preprocessing.step import (
    PolarsFrame_T0,
    PresplitStep,
)


@BaselineRegistry.preprocessing.register("age_filter")
class AgeFilter(PresplitStep):
    def __init__(self, min_age: int, max_age: int, age_col_name: str):
        self.min_age = min_age
        self.max_age = max_age
        self.age = pl.col(age_col_name)

    def apply(self, input_df: PolarsFrame_T0) -> PolarsFrame_T0:
        return input_df.filter((self.age >= self.min_age) & (self.age <= self.max_age))
