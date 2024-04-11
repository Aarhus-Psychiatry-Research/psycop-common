import pandas as pd
import polars as pl

from psycop.common.cohort_definition import PredictionTimeFilter
from psycop.common.feature_generation.loaders.raw.load_diagnoses import (
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


class  BipolarPatientsWithF20F25Filter(PredictionTimeFilter):
    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        f20_df = schizophrenia()
        f25_df = schizoaffective()
        df = df.collect().to_pandas() # type: ignore

        merged_df_f20 = pd.merge(df, f20_df, on='dw_ek_borger', how='left', suffixes=('_df', '_f20')) # type: ignore
        bipolar_patients_with_later_f20 = merged_df_f20[merged_df_f20['timestamp_df'] <= merged_df_f20['timestamp_f20']].dw_ek_borger.unique()
        
        merged_df_f25 = pd.merge(df, f25_df, on='dw_ek_borger', how='left', suffixes=('_df', '_f25')) # type: ignore
        bipolar_patients_with_later_f25 = merged_df_f25[merged_df_f25['timestamp_df'] <= merged_df_f25['timestamp_f25']].dw_ek_borger.unique()
        
        return df
