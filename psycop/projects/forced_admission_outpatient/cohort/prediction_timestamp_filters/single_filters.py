import polars as pl

from psycop.common.cohort_definition import PredictionTimeFilter
from psycop.common.feature_generation.loaders.raw.load_moves import MoveIntoRMBaselineLoader
from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_other import (
    QuarantineFilter,
)
from psycop.projects.forced_admission_outpatient.cohort.add_age import add_age
from psycop.projects.forced_admission_outpatient.cohort.extract_admissions_and_visits.get_forced_admissions import (
    forced_admissions_end_timestamps,
)
from psycop.projects.forced_admission_outpatient.cohort.prediction_timestamp_filters.eligible_config import (
    AGE_COL_NAME,
    MIN_AGE,
    MIN_DATE,
)


class ForcedAdmissionsOutpatientMinDateFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        after_df = df.filter(pl.col("timestamp") > MIN_DATE)
        return after_df


class ForcedAdmissionsOutpatientMinAgeFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        df = add_age(df.collect()).lazy()
        after_df = df.filter(pl.col(AGE_COL_NAME) >= MIN_AGE)
        return after_df


class ForcedAdmissionsOutpatientWashoutMove(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        not_within_a_year_from_move = QuarantineFilter(
            entity_id_col_name="dw_ek_borger",
            quarantine_timestamps_loader=MoveIntoRMBaselineLoader(),
            quarantine_interval_days=365,
            timestamp_col_name="timestamp",
        ).apply(df)

        return not_within_a_year_from_move


class ForcedAdmissionsEndTimestampsLoader(BaselineDataLoader):
    def load(self) -> pl.LazyFrame:
        return pl.from_pandas(forced_admissions_end_timestamps()).lazy()


class ForcedAdmissionsOutpatientWashoutPriorForcedAdmission(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        not_within_two_years_from_forced_admission = QuarantineFilter(
            entity_id_col_name="dw_ek_borger",
            quarantine_timestamps_loader=ForcedAdmissionsEndTimestampsLoader(),
            quarantine_interval_days=730,
            timestamp_col_name="timestamp",
        ).apply(df)

        return not_within_two_years_from_forced_admission
