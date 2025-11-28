import re

import pandas as pd
import polars as pl

from psycop.common.feature_generation.loaders.raw import sql_load
from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays
from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_split import (
    FilterByOutcomeStratifiedSplits,
    RegionalFilter,
)
from psycop.projects.restraint.feature_generation.modules.loaders.load_restraint_prediction_timestamps import (
    load_restraint_prediction_timestamps,
)
from psycop.projects.t2d.paper_outputs.dataset_description.table_one.table_one_lib import (
    RowSpecification,
)


def log_cross_val_eval_df_from_best_run(experiment_name: str):
    best_run_cfg = (
        MlflowClientWrapper()
        .get_best_run_from_experiment(experiment_name=experiment_name, metric="all_oof_BinaryAUROC")
        .get_config()
    )

    train_baseline_model_from_cfg(best_run_cfg)


def read_eval_df_from_disk(experiment_path: str) -> pl.DataFrame:
    return pl.read_parquet(experiment_path + "/eval_df.parquet")


def get_split_by_id(pred_times: pl.DataFrame) -> pl.DataFrame:
    train_filter = RegionalFilter(splits_to_keep=["train", "val"])
    test_filter = RegionalFilter(splits_to_keep=["test"])

    train_data = (
        train_filter.apply(pred_times.lazy()).with_columns(pl.lit("train").alias("split")).collect()
    )
    test_data = (
        test_filter.apply(pred_times.lazy()).with_columns(pl.lit("test").alias("split")).collect()
    )

    df = pl.concat([train_data, test_data], how="vertical")
    any_id_in_multiple_splits = (
        df.group_by("dw_ek_borger")
        .agg(pl.col("split").n_unique().alias("n_splits_per_id"))
        .filter(pl.col("n_splits_per_id") > 1)
    )
    if not any_id_in_multiple_splits.is_empty():
        raise ValueError("Some patients are in multiple splits.")

    df = df.group_by("dw_ek_borger").agg(pl.col("split").first())
    return df


def add_regional_split(pred_times: pl.DataFrame) -> pl.DataFrame:
    train_filter = RegionalFilter(splits_to_keep=["train", "val"])
    test_filter = RegionalFilter(splits_to_keep=["test"])

    train_data = (
        train_filter.apply(pred_times.lazy()).with_columns(pl.lit("train").alias("split")).collect()
    )
    test_data = (
        test_filter.apply(pred_times.lazy()).with_columns(pl.lit("test").alias("split")).collect()
    )

    return pl.concat([train_data, test_data], how="vertical")


def add_stratified_split(pred_times: pl.DataFrame) -> pl.DataFrame:
    train_filter = FilterByOutcomeStratifiedSplits(splits_to_keep=["train", "val"])
    test_filter = FilterByOutcomeStratifiedSplits(splits_to_keep=["test"])

    train_data = (
        train_filter.apply(pred_times.lazy()).with_columns(pl.lit("train").alias("split")).collect()
    )
    test_data = (
        test_filter.apply(pred_times.lazy()).with_columns(pl.lit("test").alias("split")).collect()
    )

    return pl.concat([train_data, test_data], how="vertical")


def parse_timestamp_from_uuid(df: pl.DataFrame, output_col_name: str = "timestamp") -> pl.DataFrame:
    return df.with_columns(
        pl.col("pred_time_uuid")
        .str.split("-")
        .list.slice(1)
        .list.join("-")
        .str.strptime(pl.Datetime, format="%Y-%m-%d-%H-%M-%S")
        .alias(output_col_name)
    )


def parse_dw_ek_borger_from_uuid(
    df: pl.DataFrame, output_col_name: str = "dw_ek_borger"
) -> pl.DataFrame:
    return df.with_columns(
        pl.col("pred_time_uuid").str.split("-").list.first().cast(pl.Int64).alias(output_col_name)
    )


def parse_outcome_timestamps(
    df: pl.DataFrame, flattened_df_path: str, outcome_timestamp_col_name: str | None = None
) -> pl.DataFrame:
    outcome_timestamp_col_name = (
        outcome_timestamp_col_name
        if outcome_timestamp_col_name is not None
        else "outcome_timestamp"
    )

    outcome_timestamps = (
        pl.read_parquet(flattened_df_path)
        .select(["dw_ek_borger", "timestamp", outcome_timestamp_col_name])
        .rename({outcome_timestamp_col_name: "outcome_timestamp"})
    )

    eval_dataset = df.join(outcome_timestamps, on=["dw_ek_borger", "timestamp"], how="left")

    return eval_dataset


def add_age(df: pl.DataFrame, birthdays: pl.DataFrame, age_col_name: str = "age") -> pl.DataFrame:
    df = df.join(birthdays, on="dw_ek_borger", how="left")
    df = df.with_columns(
        ((pl.col("timestamp") - pl.col("date_of_birth")).dt.total_days()).alias(age_col_name)
    )
    df = df.with_columns((pl.col(age_col_name) / 365.25).alias(age_col_name))

    return df


def expand_eval_df_with_extra_cols(
    eval_df: pl.DataFrame, flattened_df_path: str, outcome_timestamp_col_name: str | None = None
) -> pd.DataFrame:
    birthdates = pl.from_pandas(birthdays())

    eval_df = parse_timestamp_from_uuid(eval_df)
    eval_df = parse_dw_ek_borger_from_uuid(eval_df)
    eval_df = add_age(eval_df, birthdates)
    eval_df = parse_outcome_timestamps(eval_df, flattened_df_path, outcome_timestamp_col_name)

    return eval_df.to_pandas()
