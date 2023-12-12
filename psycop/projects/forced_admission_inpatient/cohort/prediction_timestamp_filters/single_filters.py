import polars as pl

from psycop.common.cohort_definition import PredictionTimeFilter
from psycop.common.feature_generation.application_modules.filter_prediction_times import (
    PredictionTimeFilterer,
)
from psycop.common.feature_generation.loaders.raw.load_moves import (
    load_move_into_rm_for_exclusion,
)
from psycop.projects.forced_admission_inpatient.cohort.add_age import add_age
from psycop.projects.forced_admission_inpatient.cohort.extract_admissions_and_visits.get_forced_admissions import (
    forced_admissions_end_timestamps,
)
from psycop.projects.forced_admission_inpatient.cohort.prediction_timestamp_filters.eligible_config import (
    AGE_COL_NAME,
    MIN_AGE,
    MIN_DATE,
)


class ForcedAdmissionsInpatientMinDateFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        after_df = df.filter(pl.col("timestamp") > MIN_DATE)
        return after_df


class ForcedAdmissionsInpatientMinAgeFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        df = add_age(df.collect()).lazy()
        after_df = df.filter(pl.col(AGE_COL_NAME) >= MIN_AGE)
        return after_df


class ForcedAdmissionsInpatientWashoutMove(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        not_within_x_years_from_move = pl.from_pandas(
            PredictionTimeFilterer(
                prediction_times_df=df.collect().to_pandas(),
                entity_id_col_name="dw_ek_borger",
                quarantine_timestamps_df=load_move_into_rm_for_exclusion(),
                quarantine_interval_days=365,
                timestamp_col_name="timestamp",
            ).run_filter(),
        )
        return not_within_x_years_from_move.lazy()


class ForcedAdmissionsInpatientWashoutPriorForcedAdmission(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        not_within_x_years_from_forced_admission = pl.from_pandas(
            PredictionTimeFilterer(
                prediction_times_df=df.collect().to_pandas(),
                entity_id_col_name="dw_ek_borger",
                quarantine_timestamps_df=forced_admissions_end_timestamps(),
                quarantine_interval_days=730,
                timestamp_col_name="timestamp",
            ).run_filter(),
        )

        return not_within_x_years_from_forced_admission.lazy()
