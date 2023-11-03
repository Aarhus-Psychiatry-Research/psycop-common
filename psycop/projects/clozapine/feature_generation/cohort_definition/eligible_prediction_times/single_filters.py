import polars as pl

from psycop.common.cohort_definition import PredictionTimeFilter
from psycop.common.feature_generation.application_modules.filter_prediction_times import (
    PredictionTimeFilterer,
)
from psycop.common.feature_generation.loaders.raw.load_moves import (
    load_move_into_rm_for_exclusion,
)
from psycop.projects.clozapine.feature_generation.cohort_definition.eligible_prediction_times.add_age import (
    add_age,
)
from psycop.projects.clozapine.feature_generation.cohort_definition.eligible_prediction_times.eligible_config import (
    AGE_COL_NAME,
    MIN_AGE,
    MIN_DATE,
)
from psycop.projects.clozapine.feature_generation.cohort_definition.eligible_prediction_times.schizophrenia_diagnosis import (
    add_only_patients_with_schizophrenia,
)
from psycop.projects.clozapine.feature_generation.cohort_definition.outcome_specification.first_clozapine_prescription import (
    get_first_clozapine_prescription,
)


class ClozapineMinDateFilter(PredictionTimeFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        after_df = df.filter(pl.col("timestamp") > MIN_DATE)
        return after_df


class ClozapineMinAgeFilter(PredictionTimeFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        df = add_age(df)
        after_df = df.filter(pl.col(AGE_COL_NAME) >= MIN_AGE)
        return after_df


class ClozapineSchizophrenia(PredictionTimeFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        schizophrenia_df = add_only_patients_with_schizophrenia(df=df).select(
            pl.col("timestamp").alias("timestamp_schizophrenia"),
            pl.col("dw_ek_borger"),
        )

        prediction_times_with_schizophrenia_df = df.join(
            schizophrenia_df,
            on="dw_ek_borger",
            how="inner",
        )

        days_in_10_years = 10 * 365.25

        valid_timestamp_schizophrenia = prediction_times_with_schizophrenia_df.filter(
            (pl.col("timestamp_schizophrenia") <= pl.col("timestamp"))
            & (
                pl.col("timestamp_schizophrenia")
                >= (pl.col("timestamp") - pl.duration(days=days_in_10_years))
            ),
        ).select(
            ["dw_ek_borger", "timestamp"],
        )

        # Now, ensure that for each 'dw_ek_borger' the 'timestamp' is unique
        # This involves grouping by 'dw_ek_borger', then aggregating the 'timestamp' and dropping duplicates
        after_df = (
            valid_timestamp_schizophrenia.groupby("dw_ek_borger")
            .agg(
                [
                    pl.col("timestamp").unique().alias("timestamp"),
                ],
            )
            .explode("timestamp")
        )

        return after_df


class ClozapineWashoutMoveFilter(PredictionTimeFilter):
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


class ClozapinePrevalentFilter(PredictionTimeFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        first_clozapine_prescription = pl.from_pandas(
            get_first_clozapine_prescription(),
        ).select(
            pl.col("timestamp").alias("timestamp_outcome"),
            pl.col("dw_ek_borger"),
        )

        prediction_times_with_outcome = df.join(
            first_clozapine_prescription,
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
