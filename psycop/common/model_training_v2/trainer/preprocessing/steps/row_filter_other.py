from datetime import timedelta
from typing import Literal

import polars as pl

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.preprocessing.step import (
    PresplitStep,
)


@BaselineRegistry.preprocessing.register("age_filter")
class AgeFilter(PresplitStep):
    def __init__(self, min_age: int, max_age: int, age_col_name: str):
        self.min_age = min_age
        self.max_age = max_age
        self.age = pl.col(age_col_name)

    def apply(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        return input_df.filter((self.age >= self.min_age) & (self.age <= self.max_age))


@BaselineRegistry.preprocessing.register("window_filter")
class WindowFilter(PresplitStep):
    def __init__(
        self,
        n_days: int,
        direction: Literal["ahead", "behind"],
        timestamp_col_name: str,
    ):
        self.n_days = timedelta(n_days)
        self.direction = direction
        self.timestamp_col_name = timestamp_col_name

    def apply(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        base_column = input_df.select(pl.col(self.timestamp_col_name))

        if self.direction == "ahead":
            max_timestamp = base_column.max().collect().item()
            future_cutoff = max_timestamp - self.n_days
            input_df = input_df.filter(pl.col(self.timestamp_col_name) < future_cutoff)
        elif self.direction == "behind":
            min_timestamp = base_column.min().collect().item()
            past_cutoff = min_timestamp + self.n_days
            input_df = input_df.filter(pl.col(self.timestamp_col_name) > past_cutoff)

        return input_df
