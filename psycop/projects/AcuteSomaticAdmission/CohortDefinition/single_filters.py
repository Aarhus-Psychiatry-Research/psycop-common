import polars as pl

from psycop.common.cohort_definition import PredictionTimeFilter
from psycop.common.feature_generation.loaders.raw.load_moves import MoveIntoRMBaselineLoader
from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_other import (
    QuarantineFilter,
)
from psycop.projects.AcuteSomaticAdmission.CohortDefinition.add_age import add_age
from psycop.projects.AcuteSomaticAdmission.CohortDefinition.eligible_config import (
    AGE_COL_NAME,
    MIN_AGE,
    MIN_DATE,
)
from psycop.projects.AcuteSomaticAdmission.CohortDefinition.get_somatic_emergency_visits import (
    get_contacts_to_somatic_emergency,
)


class SomaticAdmissionMinDateFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        after_df = df.filter(pl.col("timestamp") > MIN_DATE)

        # print data om dataframe
        pl_df = after_df.collect()
        pd_df = pl_df.to_pandas()
        n_patients = pd_df["dw_ek_borger"].nunique()
        print(
            f"Antal unikke ID'er der har mindst én psykiatrisk ambulant kontakt efter 2014: {n_patients}"
        )
        antal_kontakter = pd_df.shape[0]
        print(f"Antal psykiatriske ambulante kontakter efter 2014: {antal_kontakter}")

        return after_df


class SomaticAdmissionMinAgeFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        df = add_age(df.collect()).lazy()
        after_df = df.filter(pl.col(AGE_COL_NAME) >= MIN_AGE)

        # print data om dataframe
        pl_df = after_df.collect()
        pd_df = pl_df.to_pandas()
        n_patients = pd_df["dw_ek_borger"].nunique()
        print(
            f"Antal unikke ID'er der har mindst én kontakt hvor de er ældre end 18 år: {n_patients}"
        )
        antal_kontakter = pd_df.shape[0]
        print(f"Antal kontakter hvor patienten er ældre end 18 år: {antal_kontakter}")

        return after_df


class SomaticAdmissionWashoutMove(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        not_within_a_year_from_move = QuarantineFilter(
            entity_id_col_name="dw_ek_borger",
            quarantine_timestamps_loader=MoveIntoRMBaselineLoader(),
            quarantine_interval_days=365,
            timestamp_col_name="timestamp",
        ).apply(df)

        # print data om dataframe
        pl_df = not_within_a_year_from_move.collect()
        pd_df = pl_df.to_pandas()
        n_patients = pd_df["dw_ek_borger"].nunique()
        print(
            f"Antal unikke ID'er der har mindst én kontakt hvor de ikke er flyttet til RM inden for det sidste år: {n_patients}"
        )
        antal_kontakter = pd_df.shape[0]
        print(
            f"Antal kontakter hvor patienten ikke er flyttet til RM inden for det sidste år: {antal_kontakter}"
        )

        return not_within_a_year_from_move


class SomaticAdmissionTimestampsLoader(BaselineDataLoader):
    def load(self) -> pl.LazyFrame:
        return pl.from_pandas(get_contacts_to_somatic_emergency()).lazy()


class SomaticAdmissionWashoutPriorSomaticAdmission(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        not_within_two_years_from_acute_somatic_contact = QuarantineFilter(
            entity_id_col_name="dw_ek_borger",
            quarantine_timestamps_loader=SomaticAdmissionTimestampsLoader(),
            quarantine_interval_days=365,
            timestamp_col_name="timestamp",
        ).apply(df)

        # print data om dataframe
        pl_df = not_within_two_years_from_acute_somatic_contact.collect()
        pd_df = pl_df.to_pandas()
        n_patients = pd_df["dw_ek_borger"].nunique()
        print(
            f"Antal unikke ID'er der har mindst én kontakt hvor de ikke har været indlagt akut i somatikken inden for de sidste 2 år: {n_patients}"
        )
        antal_kontakter = pd_df.shape[0]
        print(
            f"Antal kontakter hvor patienten ikke ikke har været indlagt akut i somatikken inden for de sidste 2 år: {antal_kontakter}"
        )

        return not_within_two_years_from_acute_somatic_contact
