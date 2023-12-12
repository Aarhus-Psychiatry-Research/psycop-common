import polars as pl

from psycop.common.cohort_definition import PredictionTimeFilter
from psycop.common.feature_generation.application_modules.filter_prediction_times import (
    PredictionTimeFilterer,
)
from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays
from psycop.common.feature_generation.loaders.raw.load_moves import (
    load_move_into_rm_for_exclusion,
)
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.scz_bp_eligible_config import (
    AGE_COL_NAME,
    MAX_AGE,
    MIN_AGE,
    MIN_DATE,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.first_scz_or_bp_diagnosis import (
    get_first_scz_or_bp_diagnosis,
    get_scz_bp_patients_excluded_by_washin,
)

from .....common.types.polarsframe import PolarsFrameGeneric


class SczBpMinDateFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df.filter(pl.col("timestamp") > MIN_DATE)


class SczBpMinAgeFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df.filter(pl.col(AGE_COL_NAME) >= MIN_AGE)


class SczBpMaxAgeFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df.filter(pl.col(AGE_COL_NAME) <= MAX_AGE)


class SczBpAddAge(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        birthday_df = pl.from_pandas(birthdays()).lazy()

        df = df.join(birthday_df, on="dw_ek_borger", how="inner")
        df = df.with_columns(
            ((pl.col("timestamp") - pl.col("date_of_birth")).dt.days()).alias(
                AGE_COL_NAME,
            ),
        )
        df = df.with_columns((pl.col(AGE_COL_NAME) / 365.25).alias(AGE_COL_NAME))
        return df


class SczBpWashoutMoveFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        not_within_two_years_from_move = pl.from_pandas(
            PredictionTimeFilterer(
                prediction_times_df=df.collect().to_pandas(),
                entity_id_col_name="dw_ek_borger",
                quarantine_timestamps_df=load_move_into_rm_for_exclusion(),
                quarantine_interval_days=730,
                timestamp_col_name="timestamp",
            ).run_filter(),
        )
        return not_within_two_years_from_move.lazy()


class SczBpPrevalentFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Filter prediction times where the patient already has a diagnosis of
        scizophrenia or bipolar disorder"""
        time_of_first_scz_bp_diagnosis = get_first_scz_or_bp_diagnosis().select(
            pl.col("timestamp").alias("timestamp_outcome"),
            pl.col("dw_ek_borger"),
        )

        prediction_times_with_outcome = df.join(
            time_of_first_scz_bp_diagnosis.lazy(),
            on="dw_ek_borger",
            how="inner",
        )

        prevalent_prediction_times = prediction_times_with_outcome.filter(
            pl.col("timestamp") > pl.col("timestamp_outcome"),
        )

        return df.join(
            prevalent_prediction_times,
            on=["dw_ek_borger", "timestamp"],
            how="anti",
        )


class SczBpExcludedByWashinFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        ids_to_exclude = get_scz_bp_patients_excluded_by_washin()
        return df.filter(~pl.col("dw_ek_borger").is_in(ids_to_exclude))
