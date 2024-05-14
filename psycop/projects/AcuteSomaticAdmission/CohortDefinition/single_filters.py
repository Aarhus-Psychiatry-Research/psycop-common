import polars as pl

from psycop.common.cohort_definition import PredictionTimeFilter
from psycop.common.feature_generation.loaders.raw.load_moves import MoveIntoRMBaselineLoader
from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_other import (
    QuarantineFilter,
)
from psycop.projects.AcuteSomaticAdmission.CohortDefinition.add_age import add_age
from psycop.projects.AcuteSomaticAdmission.CohortDefinition.get_somatic_emergency_visits import (
    get_contacts_to_somatic_emergency,
)
from psycop.projects.AcuteSomaticAdmission.CohortDefinition.eligible_config import (
    AGE_COL_NAME,
    MIN_AGE,
    MIN_DATE,
)


class SomaticAdmissionMinDateFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        after_df = df.filter(pl.col("timestamp") > MIN_DATE)
        return after_df


class SomaticAdmissionMinAgeFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        df = add_age(df.collect()).lazy()
        after_df = df.filter(pl.col(AGE_COL_NAME) >= MIN_AGE)
        return after_df


class SomaticAdmissionWashoutMove(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        not_within_a_year_from_move = QuarantineFilter(
            entity_id_col_name="dw_ek_borger",
            quarantine_timestamps_loader=MoveIntoRMBaselineLoader(),
            quarantine_interval_days=365,
            timestamp_col_name="timestamp",
        ).apply(df)

        return not_within_a_year_from_move

#Jeg kan være i tvivl om jeg skal bruge nedenstående
class SomaticAdmissionTimestampsLoader(BaselineDataLoader):
    def load(self) -> pl.LazyFrame:
        return pl.from_pandas(get_contacts_to_somatic_emergency()).lazy()


class SomaticAdmissionWashoutPriorSomaticAdmission(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        not_within_two_years_from_acute_somatic_contact = QuarantineFilter(
            entity_id_col_name="dw_ek_borger",
            quarantine_timestamps_loader=SomaticAdmissionTimestampsLoader(),
            quarantine_interval_days=730,
            timestamp_col_name="timestamp",
        ).apply(df)

        return not_within_two_years_from_acute_somatic_contact
