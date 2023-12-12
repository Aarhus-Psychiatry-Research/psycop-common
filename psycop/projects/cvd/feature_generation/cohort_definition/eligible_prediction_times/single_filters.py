import polars as pl

from psycop.common.cohort_definition import EagerFilter
from psycop.common.feature_generation.application_modules.filter_prediction_times import (
    PredictionTimeFilterer,
)
from psycop.common.feature_generation.loaders.raw.load_moves import (
    load_move_into_rm_for_exclusion,
)
from psycop.projects.cvd.feature_generation.cohort_definition.eligible_prediction_times.eligible_config import (
    AGE_COL_NAME,
    MIN_AGE,
    MIN_DATE,
)
from psycop.projects.cvd.feature_generation.cohort_definition.outcome_specification.combined import (
    get_first_cvd_indicator,
)
from psycop.projects.t2d.feature_generation.cohort_definition.eligible_prediction_times.add_age import (
    add_age,
)


class CVDMinDateFilter(EagerFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        after_df = df.filter(pl.col("timestamp") > MIN_DATE)
        return after_df


class CVDMinAgeFilter(EagerFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        df = add_age(df)
        after_df = df.filter(pl.col(AGE_COL_NAME) >= MIN_AGE)
        return after_df


class WithoutPrevalentCVD(EagerFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        first_cvd_indicator = pl.from_pandas(get_first_cvd_indicator())

        indicator_before_min_date = first_cvd_indicator.filter(
            pl.col("timestamp") < MIN_DATE,
        )

        prediction_times_from_patients_with_cvd = df.join(
            indicator_before_min_date,
            on="dw_ek_borger",
            how="inner",
        )

        no_prevalent_cvd = df.join(
            prediction_times_from_patients_with_cvd,
            on="dw_ek_borger",
            how="anti",
        )

        return no_prevalent_cvd.drop(["age"])


class NoIncidentCVD(EagerFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        contacts_with_cvd = pl.from_pandas(
            get_first_cvd_indicator(),
        )

        contacts_with_cvd = df.join(
            contacts_with_cvd,
            on="dw_ek_borger",
            how="left",
            suffix="_result",
        )

        after_incident_cvd = contacts_with_cvd.filter(
            pl.col("timestamp") > pl.col("timestamp_result"),
        )

        not_after_incident_cvd = contacts_with_cvd.join(
            after_incident_cvd,
            on="dw_ek_borger",
            how="anti",
        )

        return not_after_incident_cvd.drop(["timestamp_result", "cause"])


class CVDWashoutMove(EagerFilter):
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
