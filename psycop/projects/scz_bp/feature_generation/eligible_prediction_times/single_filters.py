import polars as pl

from psycop.common.cohort_definition import PredictionTimeFilter
from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays
from psycop.common.feature_generation.loaders.raw.load_moves import (
    MoveIntoRMBaselineLoader,
    load_move_into_rm_for_exclusion,
)
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_other import (
    QuarantineFilter,
)
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.scz_bp_eligible_config import (
    AGE_COL_NAME,
    MAX_AGE,
    MIN_AGE,
    MIN_DATE,
    N_DAYS_WASHIN,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.add_time_from_first_visit import (
    add_time_from_first_contact_to_psychiatry,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.first_scz_or_bp_diagnosis import (
    get_first_scz_or_bp_diagnosis,
)


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
        not_within_90_days_from_move = QuarantineFilter(
            entity_id_col_name="dw_ek_borger",
            quarantine_timestamps_loader=MoveIntoRMBaselineLoader(),
            quarantine_interval_days=N_DAYS_WASHIN,
            timestamp_col_name="timestamp",
        ).apply(df)

        return not_within_90_days_from_move


class SczBpPrevalentFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Filter prediction times occuring after a patient has received a diagnosis
        of bp or scz"""
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


class SczBpTimeFromFirstVisitFilter(PredictionTimeFilter):
    """Remove prediction times within 90 days of first psychiatric contact"""

    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        df_with_time_from_first_contact = add_time_from_first_contact_to_psychiatry(
            df=df.collect(),
        )

        return (
            df_with_time_from_first_contact.filter(
                pl.col("time_from_first_contact") > pl.duration(days=N_DAYS_WASHIN),
            )
            .select(pl.exclude(["first_contact", "time_from_first_contact"]))
            .lazy()
        )
