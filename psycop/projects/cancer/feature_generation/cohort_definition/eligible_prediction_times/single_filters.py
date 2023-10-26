import polars as pl

from psycop.common.cohort_definition import PredictionTimeFilter
from psycop.common.feature_generation.application_modules.filter_prediction_times import (
    PredictionTimeFilterer,
)
from psycop.common.feature_generation.loaders.raw.load_moves import (
    load_move_into_rm_for_exclusion,
)
from psycop.projects.cancer.feature_generation.cohort_definition.add_age import (
    add_age,
)
from psycop.projects.cancer.feature_generation.cohort_definition.eligible_prediction_times.cancer_eligible_config import (
    AGE_COL_NAME,
    MIN_AGE,
    MIN_DATE,
)
from psycop.projects.cancer.feature_generation.cohort_definition.outcome_specification.cancer_outcomes import (
    get_first_cancer_diagnosis,
)


class CancerMinDateFilter(PredictionTimeFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("timestamp") > MIN_DATE)


class CancerMinAgeFilter(PredictionTimeFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        df = add_age(df)
        return df.filter(pl.col(AGE_COL_NAME) >= MIN_AGE)


class CancerWashoutMoveFilter(PredictionTimeFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        not_within_half_a_year_from_move = pl.from_pandas(
            PredictionTimeFilterer(
                prediction_times_df=df.to_pandas(),
                entity_id_col_name="dw_ek_borger",
                quarantine_timestamps_df=load_move_into_rm_for_exclusion(),
                quarantine_interval_days=182,
                timestamp_col_name="timestamp",
            ).run_filter(),
        )

        return not_within_half_a_year_from_move


class CancerPrevalentFilter(PredictionTimeFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        first_cancer_diagnosis = pl.from_pandas(get_first_cancer_diagnosis()).select(
            pl.col("timestamp").alias("timestamp_outcome"),
            pl.col("dw_ek_borger"),
        )

        prediction_times_with_outcome = df.join(
            first_cancer_diagnosis,
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
