import polars as pl

from psycop.common.cohort_definition import PredictionTimeFilter
from psycop.common.feature_generation.loaders.raw.load_moves import MoveIntoRMBaselineLoader
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_other import (
    QuarantineFilter,
)
from psycop.projects.psychometrics.mania_rating_scales.cohort.eligible_prediction_times.add_age import (
    add_age,
)
from psycop.projects.psychometrics.mania_rating_scales.cohort.eligible_prediction_times.eligible_config import (
    AGE_COL_NAME,
    MIN_AGE,
    MIN_DATE,
)
from psycop.projects.psychometrics.mania_rating_scales.cohort.eligible_prediction_times.manic_and_bipolar_disorders import (
    add_only_patients_with_manic_and_bipolar_diagnosis,
)


class PsychometricsMinDateFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        after_df = df.filter(pl.col("timestamp") > MIN_DATE)
        return after_df


class PsychometricsMinAgeFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        df = add_age(df.collect()).lazy()
        after_df = df.filter(pl.col(AGE_COL_NAME) >= MIN_AGE)
        return after_df


class ManiaRatingBipolarDisorders(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        bipolar_disorders_df = (
            add_only_patients_with_manic_and_bipolar_diagnosis()
            .lazy()
            .select(pl.col("timestamp").alias("timestamp_bipolar"), pl.col("dw_ek_borger"))
        )

        prediction_times_with_bipolar_disorders_df = df.join(
            bipolar_disorders_df, on="dw_ek_borger", how="inner"
        )

        days_in_1_year = 1 * 365

        valid_timestamp_bipolar_disorder = prediction_times_with_bipolar_disorders_df.filter(
            (pl.col("timestamp_bipolar") <= pl.col("timestamp"))
            & (
                pl.col("timestamp_bipolar")
                >= (pl.col("timestamp") - pl.duration(days=days_in_1_year))
            )
        ).select(["dw_ek_borger", "timestamp"])

        # Now, ensure that for each 'dw_ek_borger' the 'timestamp' is unique
        # This involves grouping by 'dw_ek_borger', then aggregating the 'timestamp' and dropping duplicates
        after_df = (
            valid_timestamp_bipolar_disorder.groupby("dw_ek_borger")
            .agg([pl.col("timestamp").unique().alias("timestamp")])
            .explode("timestamp")
        )

        return after_df


class ManiaRatingWashoutMoveFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        not_within_half_a_year_from_move = QuarantineFilter(
            entity_id_col_name="dw_ek_borger",
            quarantine_timestamps_loader=MoveIntoRMBaselineLoader(),
            quarantine_interval_days=30,
            timestamp_col_name="timestamp",
        ).apply(df)

        return not_within_half_a_year_from_move
