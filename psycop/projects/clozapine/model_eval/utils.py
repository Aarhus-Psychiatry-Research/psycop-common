import pandas as pd
import polars as pl

from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays
from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg


def log_cross_val_eval_df_from_best_run(experiment_name: str):
    best_run_cfg = (
        MlflowClientWrapper()
        .get_best_run_from_experiment(experiment_name=experiment_name, metric="all_oof_BinaryAUROC")
        .get_config()
    )

    train_baseline_model_from_cfg(best_run_cfg)


def read_eval_df_from_disk(experiment_path: str) -> pl.DataFrame:
    return pl.read_parquet(experiment_path + "/eval_df.parquet")


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
        .with_columns(pl.col("timestamp").dt.cast_time_unit("us"))
        .select(["dw_ek_borger", "timestamp", outcome_timestamp_col_name])
        .rename({outcome_timestamp_col_name: "timestamp_outcome"})
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
