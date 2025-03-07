import polars as pl

from psycop.common.cohort_definition import PredictionTimeFilter
from psycop.common.feature_generation.loaders.raw.load_moves import MoveIntoRMBaselineLoader
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_other import (
    QuarantineFilter,
)
from psycop.projects.LungCancer.cohort_definition.eligible_prediction_times.add_age import add_age
from psycop.projects.LungCancer.cohort_definition.eligible_prediction_times.lung_cancer_eligible_config import (
    AGE_COL_NAME,
    MIN_AGE,
    MIN_DATE,
)
from psycop.projects.LungCancer.cohort_definition.outcome_specification.first_lung_cancer_diagnosis import (
    get_first_lung_cancer_diagnosis,
)


class LungCancerMinDateFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df.filter(pl.col("timestamp") > MIN_DATE)


class LungCancerMinAgeFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        df = add_age(df.collect()).lazy()
        return df.filter(pl.col(AGE_COL_NAME) >= MIN_AGE)


class LungCancerWashoutMoveFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        not_within_half_a_year_from_move = QuarantineFilter(
            entity_id_col_name="dw_ek_borger",
            quarantine_timestamps_loader=MoveIntoRMBaselineLoader(),
            quarantine_interval_days=182,
            timestamp_col_name="timestamp",
        ).apply(df)

        return not_within_half_a_year_from_move


class LungCancerPrevalentFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        first_lung_cancer_diagnosis = pl.from_pandas(get_first_lung_cancer_diagnosis()).select(
            pl.col("timestamp").alias("timestamp_outcome"), pl.col("dw_ek_borger")
        )

        prediction_times_with_outcome = df.join(
            first_lung_cancer_diagnosis.lazy(), on="dw_ek_borger", how="inner"
        )

        prevalent_prediction_times = prediction_times_with_outcome.filter(
            pl.col("timestamp") > pl.col("timestamp_outcome")
        )

        return df.join(prevalent_prediction_times, on=["dw_ek_borger", "timestamp"], how="anti")
