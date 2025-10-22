from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

import polars as pl

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.preprocessing.step import PresplitStep

from ...base_dataloader import BaselineDataLoader


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
    """Filter rows by whether their timestamp is within a certain window of the min or max timestamp in the dataset."""

    def __init__(self, n_days: int, direction: Literal["ahead", "behind"], timestamp_col_name: str):
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
@dataclass
class QuarantineFilter(PresplitStep):
    entity_id_col_name: str
    timestamp_col_name: str
    quarantine_timestamps_loader: BaselineDataLoader
    quarantine_interval_days: int
    validate_on_init: bool = True
    _tmp_pred_time_uuid_col_name = "_tmp_pred_time_uuid"

    def __post_init__(self) -> None:
        required_columns = [self.timestamp_col_name, self.entity_id_col_name]
        if self.validate_on_init and not all(
            col in self.quarantine_timestamps_loader.load().columns for col in required_columns
        ):
            raise ValueError(
                "The quarantine timestamps loader must load a dataframe with the columns 'timestamp' and "
                f"'{self.entity_id_col_name}'"
            )

    def _generate_pred_time_uuid_column(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        input_df = input_df.with_columns(
            pl.concat_str(
                [
                    pl.col(self.entity_id_col_name).cast(pl.Utf8),
                    pl.lit("-"),
                    pl.col(self.timestamp_col_name).dt.strftime("%Y-%m-%d-%H-%M-%S"),
                ]
            ).alias(self._tmp_pred_time_uuid_col_name)
        )

        return input_df

    def apply(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        # We need to check if ANY quarantine date hits each prediction time.
        # Create combinations

        added_pred_time_uuid_col = False
        quarantine_timestamps_df = self.quarantine_timestamps_loader.load()

        if self._tmp_pred_time_uuid_col_name not in input_df.columns:
            input_df = self._generate_pred_time_uuid_column(input_df)
            added_pred_time_uuid_col = True
        if self._tmp_pred_time_uuid_col_name not in quarantine_timestamps_df.columns:
            quarantine_timestamps_df = self._generate_pred_time_uuid_column(
                quarantine_timestamps_df
            )

        df_with_quarantine_timestamps = input_df.join(
            quarantine_timestamps_df.rename({self.timestamp_col_name: "timestamp_quarantine"}),
            on=self.entity_id_col_name,
            how="left",
        ).with_columns(
            pred_time_uuid=pl.col(self.timestamp_col_name).dt.strftime("%Y-%m-%d-%H-%M-%S")
        )

        time_since_quarantine = df_with_quarantine_timestamps.with_columns(
            (pl.col(self.timestamp_col_name) - pl.col("timestamp_quarantine"))
            .dt.seconds()
            .alias("seconds_since_quarantine")
        )

        # Check if the quarantine date hits the prediction time
        hit_by_quarantine = time_since_quarantine.filter(
            (pl.col("seconds_since_quarantine") < self.quarantine_interval_days * 60 * 60 * 24)
            & (pl.col("seconds_since_quarantine") > 0)
        ).select(self._tmp_pred_time_uuid_col_name)

        # Use these rows to filter the prediction times, ensuring that all columns are kept
        df = input_df.join(hit_by_quarantine, on=self._tmp_pred_time_uuid_col_name, how="anti")

        if added_pred_time_uuid_col:
            df = df.drop(self._tmp_pred_time_uuid_col_name)

        return df


@BaselineRegistry.preprocessing.register("date_filter")
@dataclass
class DateFilter(PresplitStep):
    """Filter rows based on a date column and a threshold date."""

    column_name: str
    threshold_date: str
    direction: Literal["before", "after-inclusive"]

    def apply(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        date_col = pl.col(self.column_name)
        threshold_date = pl.lit(self.threshold_date).cast(pl.Date)

        match self.direction:
            case "before":
                return input_df.filter(date_col < threshold_date)
            case "after-inclusive":
                return input_df.filter(date_col >= threshold_date)


@BaselineRegistry.preprocessing.register("value_filter")
@dataclass
class ValueFilter(PresplitStep):
    """Filter rows based on a column and a value."""

    column_name: str
    threshold_value: float
    direction: Literal["before", "after-inclusive"]

    def apply(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        value_col = pl.col(self.column_name)

        match self.direction:
            case "before":
                return input_df.filter(value_col < self.threshold_value)
            case "after-inclusive":
                return input_df.filter(value_col >= self.threshold_value)


@BaselineRegistry.preprocessing.register("cooldown_after_positive_filter")
@dataclass
class CooldownAfterPositiveFilter(PresplitStep):
    """
    Filter rows by removing the next `n_cooldown_days` after a positive outcome
    in each group defined by `group_by_cols`.
    """

    n_cooldown_days: int
    group_by_cols: Sequence[str]
    timestamp_col_name: str
    outcome_col_name: str

    def apply(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Keep rows in each group until and after the positive outcome,
        then remove the next `n_cooldown_days` rows after each positive.
        """

        # Step 1: Sort rows within each group by timestamp
        df = input_df.sort([*self.group_by_cols, self.timestamp_col_name])  # type: ignore

        # Step 2: Add helper columns
        df = df.with_columns(
            [
                (pl.col(self.outcome_col_name) == 1).cast(pl.Int64).alias("event_flag"),
                pl.arange(0, pl.count(), 1).over(self.group_by_cols).alias("row_idx"),
            ]
        )

        # Step 3: Define the per-group cooldown marking function
        def mark_cooldown(group: pl.DataFrame) -> pl.DataFrame:
            event_rows = group.filter(pl.col("event_flag") == 1)["row_idx"].to_list()
            mask = [False] * len(group)
            for r in event_rows:
                for offset in range(1, self.n_cooldown_days + 1):
                    if r + offset < len(mask):
                        mask[r + offset] = True
            # Keep only rows NOT in cooldown
            return group.filter(~pl.Series(mask))

        schema: dict[str, pl.DataType] = dict(input_df.schema.items())

        # Step 5: Apply per-group cooldown
        filtered = df.groupby(self.group_by_cols).apply(mark_cooldown, schema=schema)

        # Step 6: Drop helper columns
        cleaned = filtered.collect().drop(["event_flag", "row_idx"])

        return cleaned.lazy()
