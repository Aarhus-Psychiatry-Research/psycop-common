import pandas as pd
import polars as pl

from psycop.common.cohort_definition import PredictionTimeFilter
from psycop.common.feature_generation.loaders.raw.load_diagnoses import (
    bipolar,
    depressive_disorders,
    schizoaffective,
    schizophrenia,
)
from psycop.common.feature_generation.loaders.raw.load_moves import (
    MoveIntoRMBaselineLoader,
)
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_other import (
    QuarantineFilter,
)
from psycop.projects.bipolar.cohort_definition.eligible_data.add_age import add_age
from psycop.projects.bipolar.cohort_definition.eligible_data.eligible_config import (
    AGE_COL_NAME,
    MIN_AGE,
    MIN_DATE,
)


class BipolarMinDateFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        after_df = df.filter(pl.col("timestamp") > MIN_DATE)
        return after_df


class BipolarMinAgeFilter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        df = add_age(df.collect()).lazy()
        after_df = df.filter(pl.col(AGE_COL_NAME) >= MIN_AGE)
        return after_df


class BipolarWashoutMove(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        not_within_two_years_from_move = QuarantineFilter(
            entity_id_col_name="dw_ek_borger",
            quarantine_timestamps_loader=MoveIntoRMBaselineLoader(),
            quarantine_interval_days=730,
            timestamp_col_name="timestamp",
        ).apply(df)

        return not_within_two_years_from_move


class BipolarPatientsWithF32F38Filter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        f32_38_df = depressive_disorders()
        pd_df = pd.DataFrame(df.collect().to_pandas())

        merged_df_f32 = pd.merge(
            pd_df, f32_38_df, on="dw_ek_borger", how="left", suffixes=("_df", "_f32")
        )
        bipolar_patients_with_earlier_f32 = merged_df_f32[
            merged_df_f32["timestamp_df"] >= merged_df_f32["timestamp_f32"]
        ].dw_ek_borger.unique()

        filtered_df = pd_df[pd_df["dw_ek_borger"].isin(bipolar_patients_with_earlier_f32)]

        filtered_df = pl.DataFrame(filtered_df).lazy()

        return filtered_df


class PatientsWithF20F25Filter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        f20_df = schizophrenia()
        f25_df = schizoaffective()
        pd_df = pd.DataFrame(df.collect().to_pandas())

        merged_df_f20 = pd.merge(
            pd_df, f20_df, on="dw_ek_borger", how="left", suffixes=("_df", "_f20")
        )
        bipolar_patients_with_later_f20 = merged_df_f20[
            merged_df_f20["timestamp_df"] <= merged_df_f20["timestamp_f20"]
        ].dw_ek_borger.unique()

        merged_df_f25 = pd.merge(
            pd_df, f25_df, on="dw_ek_borger", how="left", suffixes=("_df", "_f25")
        )
        bipolar_patients_with_later_f25 = merged_df_f25[
            merged_df_f25["timestamp_df"] <= merged_df_f25["timestamp_f25"]
        ].dw_ek_borger.unique()

        bipolar_patients_with_f20_f25 = set(bipolar_patients_with_later_f20).union(
            set(bipolar_patients_with_later_f25)
        )
        filtered_df = pd_df[~pd_df["dw_ek_borger"].isin(bipolar_patients_with_f20_f25)]

        filtered_df = pl.DataFrame(filtered_df).lazy()

        return filtered_df


class DepressiveDisorderPatientsWithF31Filter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        f31_df = bipolar()
        pd_df = pd.DataFrame(df.collect().to_pandas())

        filtered_df = pd_df[~pd_df["dw_ek_borger"].isin(f31_df["dw_ek_borger"])]

        filtered_df = pl.DataFrame(filtered_df).lazy()

        return filtered_df
