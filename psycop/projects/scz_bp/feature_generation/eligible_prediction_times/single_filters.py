import polars as pl

from psycop.common.cohort_definition import EagerFilter
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


class SczBpMinDateFilter(EagerFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("timestamp") > MIN_DATE)


class SczBpMinAgeFilter(EagerFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col(AGE_COL_NAME) >= MIN_AGE)


class SczBpMaxAgeFilter(EagerFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col(AGE_COL_NAME) <= MAX_AGE)


class SczBpAddAge(EagerFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        birthday_df = pl.from_pandas(birthdays())

        df = df.join(birthday_df, on="dw_ek_borger", how="inner")
        df = df.with_columns(
            ((pl.col("timestamp") - pl.col("date_of_birth")).dt.days()).alias(
                AGE_COL_NAME,
            ),
        )
        df = df.with_columns((pl.col(AGE_COL_NAME) / 365.25).alias(AGE_COL_NAME))
        return df


class SczBpWashoutMoveFilter(EagerFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        not_within_two_years_from_move = pl.from_pandas(
            PredictionTimeFilterer(
                prediction_times_df=df.to_pandas(),
                entity_id_col_name="dw_ek_borger",
                quarantine_timestamps_df=load_move_into_rm_for_exclusion(),
                quarantine_interval_days=730,
                timestamp_col_name="timestamp",
            ).run_filter(),
        )
        return not_within_two_years_from_move


class SczBpPrevalentFilter(EagerFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        """Filter prediction times where the patient already has a diagnosis of
        scizophrenia or bipolar disorder"""
        time_of_first_scz_bp_diagnosis = get_first_scz_or_bp_diagnosis().select(
            pl.col("timestamp").alias("timestamp_outcome"),
            pl.col("dw_ek_borger"),
        )

        prediction_times_with_outcome = df.join(
            time_of_first_scz_bp_diagnosis,
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


class SczBpExcludedByWashinFilter(EagerFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        ids_to_exclude = get_scz_bp_patients_excluded_by_washin()
        return df.filter(~pl.col("dw_ek_borger").is_in(ids_to_exclude))
