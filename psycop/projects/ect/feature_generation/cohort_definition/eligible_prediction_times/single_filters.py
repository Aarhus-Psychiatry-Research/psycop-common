import polars as pl

from psycop.common.cohort_definition import PredictionTimeFilter
from psycop.common.feature_generation.loaders.raw.load_diagnoses import f2_disorders, f6_disorders
from psycop.common.feature_generation.loaders.raw.load_moves import MoveIntoRMBaselineLoader
from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_other import (
    QuarantineFilter,
)
from psycop.projects.ect.feature_generation.cohort_definition.add_age import add_age
from psycop.projects.ect.feature_generation.cohort_definition.eligible_prediction_times.eligible_config import (
    AGE_COL_NAME,
    MIN_AGE,
    MIN_DATE,
)
from psycop.projects.ect.feature_generation.cohort_definition.outcome_specification.procedure_codes import (
    get_ect_procedures,
)


class ECTMinDateFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        after_df = df.filter(pl.col("timestamp") > MIN_DATE)
        return after_df


class ECTMinAgeFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        df = add_age(df.collect()).lazy()
        after_df = df.filter(pl.col(AGE_COL_NAME) >= MIN_AGE)
        return after_df.drop("age")


class ECTProcedureTimestampsLoader(BaselineDataLoader):
    def load(self) -> pl.LazyFrame:
        return get_ect_procedures().drop("procedurekodetekst").lazy()


class NoIncidentECTWithin3Years(PredictionTimeFilter):  # TODO: check this
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        not_within_3_years_from_ect = QuarantineFilter(
            entity_id_col_name="dw_ek_borger",
            quarantine_timestamps_loader=ECTProcedureTimestampsLoader(),
            quarantine_interval_days=365 * 3,
            timestamp_col_name="timestamp",
        ).apply(df)

        return not_within_3_years_from_ect


def remove_after_incidence(
    prediction_times: pl.LazyFrame, first_incidence_df: pl.LazyFrame
) -> pl.LazyFrame:
    pred_times_with_time_of_incidence = prediction_times.join(
        first_incidence_df, on="dw_ek_borger", how="left", suffix="_result"
    )

    after_incidence = pred_times_with_time_of_incidence.filter(
        pl.col("timestamp") > pl.col("timestamp_result")
    )

    not_after_incidence = pred_times_with_time_of_incidence.join(
        after_incidence, on="dw_ek_borger", how="anti"
    )

    return not_after_incidence.drop(["timestamp_result"])


class NoIncidentF2(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        first_f2 = (
            (
                pl.from_pandas(f2_disorders())
                .sort(["dw_ek_borger", "timestamp"])
                .group_by("dw_ek_borger")
                .first()
            )
            .drop("value")
            .lazy()
        )

        return remove_after_incidence(prediction_times=df, first_incidence_df=first_f2)


class NoIncidentF6(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        first_f6 = (
            (
                pl.from_pandas(f6_disorders())
                .sort(["dw_ek_borger", "timestamp"])
                .group_by("dw_ek_borger")
                .first()
            )
            .drop("value")
            .lazy()
        )

        return remove_after_incidence(prediction_times=df, first_incidence_df=first_f6)


class ECTWashoutMove(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        not_within_two_years_from_move = QuarantineFilter(
            entity_id_col_name="dw_ek_borger",
            quarantine_timestamps_loader=MoveIntoRMBaselineLoader(),
            quarantine_interval_days=730,
            timestamp_col_name="timestamp",
        ).apply(df)

        return not_within_two_years_from_move
