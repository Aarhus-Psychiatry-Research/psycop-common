import re
from typing import Literal

import pandas as pd
import polars as pl

from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays
from psycop.common.feature_generation.loaders.raw import sql_load
from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_split import (
    RegionalFilter,
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


def get_psychiatric_diagnosis_row_specs(col_names: list[str]) -> list[RowSpecification]:
    pattern = re.compile(r"pred_f\d_disorders")
    columns = sorted([c for c in col_names if pattern.search(c) and "730" in c])

    readable_col_names = []

    for c in columns:
        icd_10_fx = re.findall(r"f\d+", c)[0]
        readable_col_name = f"{icd_10_fx.capitalize()} disorders prior 2 years"
        readable_col_names.append(readable_col_name)

    specs = []
    for i, _ in enumerate(readable_col_names):
        specs.append(
            RowSpecification(
                source_col_name=columns[i],
                readable_name=readable_col_names[i],
                categorical=True,
                nonnormal=False,
                values_to_display=[1],
            )
        )

    return specs


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


def add_split(pred_times: pl.DataFrame) -> pl.DataFrame:
    train_filter = RegionalFilter(splits_to_keep=["train", "val"])
    test_filter = RegionalFilter(splits_to_keep=["test"])

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


def parse_outcome_timestamps(df: pl.DataFrame) -> pl.DataFrame:
    outcome_timestamps = pl.DataFrame(
        sql_load(
            "SELECT pred_times.dw_ek_borger, pred_time, first_mechanical_restraint as timestamp FROM fct.psycop_coercion_outcome_timestamps as pred_times LEFT JOIN fct.psycop_coercion_outcome_timestamps_2 as outc_times ON (pred_times.dw_ek_borger = outc_times.dw_ek_borger AND pred_times.datotid_start = outc_times.datotid_start)"
        ).drop_duplicates()
    )

    eval_dataset = df.with_columns(pl.col("timestamp").dt.cast_time_unit("ns")).join(
        outcome_timestamps,
        left_on=["dw_ek_borger", "timestamp"],
        right_on=["dw_ek_borger", "pred_time"],
        suffix="_outcome",
        how="left",
    )

    return eval_dataset


def add_age(df: pl.DataFrame, birthdays: pl.DataFrame, age_col_name: str = "age") -> pl.DataFrame:
    df = df.join(birthdays, on="dw_ek_borger", how="left")
    df = df.with_columns(
        ((pl.col("timestamp") - pl.col("date_of_birth")).dt.total_days()).alias(age_col_name)
    )
    df = df.with_columns((pl.col(age_col_name) / 365.25).alias(age_col_name))

    return df


def expand_eval_df_with_extra_cols(eval_df: pl.DataFrame, birthdays: pl.DataFrame) -> pd.DataFrame:
    birthdays = pl.from_pandas(birthdays())  # type: ignore

    eval_df = parse_timestamp_from_uuid(eval_df)
    eval_df = parse_dw_ek_borger_from_uuid(eval_df)
    eval_df = add_age(eval_df, birthdays)
    eval_df = parse_outcome_timestamps(eval_df)

    return eval_df.to_pandas()
