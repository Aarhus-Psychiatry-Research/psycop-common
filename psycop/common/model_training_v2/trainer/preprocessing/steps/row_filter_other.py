from dataclasses import dataclass
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


@BaselineRegistry.preprocessing.register("quarantine_filter")
@dataclass(frozen=True)
class QuarantineFilter:
    entity_id_col_name: str
    pred_time_uuid_col_name: str
    timestamp_col_name: str
    quarantine_timestamps_df: pl.LazyFrame
    quarantine_interval_days: int

    def apply(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        # We need to check if ANY quarantine date hits each prediction time.
        # Create combinations
        df = input_df.join(
            self.quarantine_timestamps_df.rename({"timestamp": "timestamp_quarantine"}),
            on=self.entity_id_col_name,
            how="left",
        )

        df = df.with_columns(
            (pl.col(self.timestamp_col_name) - pl.col("timestamp_quarantine"))
            .dt.days()
            .alias("days_since_quarantine"),
        )

        # Check if the quarantine date hits the prediction time
        df = df.with_columns(
            pl.when(
                (pl.col("days_since_quarantine") < self.quarantine_interval_days)
                & (pl.col("days_since_quarantine") > 0),
            )
            .then(True)
            .otherwise(False)
            .alias("hit_by_quarantine"),
        )

        # Get only the rows that were hit by the quarantine date
        df_hit_by_quarantine = df.unique(
            subset=[self.pred_time_uuid_col_name],
        ).select(  # noqa: E712
            ["pred_time_uuid", "hit_by_quarantine"],
        )

        # Use these rows to filter the prediction times
        df = input_df.join(
            df_hit_by_quarantine,
            on=self.pred_time_uuid_col_name,
            how="left",
            suffix=("_hit_by_quarantine"),
            validate="1:1",
        ).filter(
            pl.col("hit_by_quarantine") == False,  # noqa: E712,
        )

        # Drop the columns we added
        df = df.drop(
            columns=[
                "hit_by_quarantine",
            ],
        )
        return df
