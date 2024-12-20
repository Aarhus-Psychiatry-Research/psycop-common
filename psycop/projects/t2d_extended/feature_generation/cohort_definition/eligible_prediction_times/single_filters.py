import polars as pl
from wasabi import Printer

from psycop.common.feature_generation.loaders.raw.load_moves import MoveIntoRMBaselineLoader
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_other import (
    QuarantineFilter,
)

msg = Printer(timestamp=True)

from psycop.common.cohort_definition import PredictionTimeFilter
from psycop.projects.t2d_extended.feature_generation.cohort_definition.eligible_prediction_times.eligible_config import (
    AGE_COL_NAME,
    MIN_AGE,
    MIN_DATE,
)
from psycop.projects.t2d_extended.feature_generation.cohort_definition.outcome_specification.combined import (
    get_first_diabetes_indicator,
)
from psycop.projects.t2d_extended.feature_generation.cohort_definition.outcome_specification.lab_results import (
    get_first_diabetes_lab_result_above_threshold,
)


class T2DMinDateFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        after_df = df.filter(pl.col("timestamp") > MIN_DATE)
        return after_df


class T2DMinAgeFilter(PredictionTimeFilter):
    def __init__(self, birthday_df: pl.LazyFrame) -> None:
        self.birthday_df = birthday_df

    def _add_age(self, df: pl.LazyFrame) -> pl.LazyFrame:
        df = df.join(self.birthday_df, on="dw_ek_borger", how="inner")
        df = df.with_columns(
            ((pl.col("timestamp") - pl.col("date_of_birth")).dt.days()).alias(AGE_COL_NAME)
        )
        df = df.with_columns((pl.col(AGE_COL_NAME) / 365.25).alias(AGE_COL_NAME))

        return df

    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        df = self._add_age(df)
        after_df = df.filter(pl.col(AGE_COL_NAME) >= MIN_AGE)
        return after_df


class WithoutPrevalentDiabetes(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        first_diabetes_indicator = pl.from_pandas(get_first_diabetes_indicator()).lazy()

        indicator_before_min_date = first_diabetes_indicator.filter(pl.col("timestamp") < MIN_DATE)

        prediction_times_from_patients_with_diabetes = df.join(
            indicator_before_min_date, on="dw_ek_borger", how="inner"
        )

        no_prevalent_diabetes = df.join(
            prediction_times_from_patients_with_diabetes, on="dw_ek_borger", how="anti"
        )

        return no_prevalent_diabetes.drop(["age"])


class NoIncidentDiabetes(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        results_above_threshold = pl.from_pandas(
            get_first_diabetes_lab_result_above_threshold()
        ).lazy()

        contacts_with_hba1c = df.join(
            results_above_threshold, on="dw_ek_borger", how="left", suffix="_result"
        )

        after_incident_diabetes = contacts_with_hba1c.filter(
            pl.col("timestamp") > pl.col("timestamp_result")
        )

        not_after_incident_diabetes = contacts_with_hba1c.join(
            after_incident_diabetes, on="dw_ek_borger", how="anti"
        )

        return not_after_incident_diabetes.drop(["timestamp_result", "value"])


class T2DWashoutMove(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        msg.info("Applying filter")
        not_within_two_years_from_move = QuarantineFilter(
            entity_id_col_name="dw_ek_borger",
            quarantine_timestamps_loader=MoveIntoRMBaselineLoader(),
            quarantine_interval_days=730,
            timestamp_col_name="timestamp",
        ).apply(df)

        msg.info("Returning")
        return not_within_two_years_from_move
