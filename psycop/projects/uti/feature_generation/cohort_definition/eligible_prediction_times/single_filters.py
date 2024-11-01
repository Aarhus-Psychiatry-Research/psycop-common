from datetime import datetime

import polars as pl
from wasabi import Printer

from psycop.common.feature_generation.loaders.raw.load_moves import MoveIntoRMBaselineLoader

from ......common.model_training_v2.trainer.preprocessing.steps.row_filter_other import (
    QuarantineFilter,
)

msg = Printer(timestamp=True)

from psycop.common.cohort_definition import PredictionTimeFilter
from psycop.projects.uti.feature_generation.cohort_definition.eligible_prediction_times.eligible_config import (
    ADMISSION_TYPE,
    MIN_AGE,
    MIN_DATE,
)


class UTIExcludeFirstDayFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df.filter(pl.col("pred_adm_day_count") != 1)


class UTIAdmissionFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df.filter(
            (pl.col("datotid_slut").is_not_null())
            & (pl.col("datotid_slut") <= datetime(year=2021, month=11, day=22))
        )


class UTIAdmissionTypeFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df.filter(pl.col("pt_type") == ADMISSION_TYPE)


class UTIMinDateFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        after_df = df.filter(pl.col("timestamp") > MIN_DATE)
        return after_df


class UTIMinAgeFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df.filter(pl.col("alder_start") >= MIN_AGE)


class UTIWashoutMove(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        msg.info("Applying filter")
        not_within_two_years_from_move = QuarantineFilter(
            entity_id_col_name="dw_ek_borger",
            quarantine_timestamps_loader=MoveIntoRMBaselineLoader(),
            quarantine_interval_days=730,
            timestamp_col_name="timestamp",
        ).apply(df)

        msg.info("Returning")
        return not_within_two_years_from_move
