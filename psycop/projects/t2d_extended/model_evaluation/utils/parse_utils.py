import pandas as pd
import polars as pl

from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.common.model_training_v2.config.config_utils import PsycopConfig


def parse_dw_ek_borger_from_uuid(
    df: pl.DataFrame, output_col_name: str = "dw_ek_borger"
) -> pl.DataFrame:
    return df.with_columns(
        pl.col("pred_time_uuids").str.split("-").list.first().cast(pl.Int64).alias(output_col_name)
    )


def fix_pred_timestamps(
    df: pl.DataFrame, col: str = "pred_timestamps_joined", output_col: str = "pred_timestamps"
) -> pl.DataFrame:
    df = df.with_columns(pl.col("pred_timestamps").list.join("-").alias("pred_timestamps_joined"))

    return df.with_columns(
        pl.col(col)
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f", strict=False)
        .alias(output_col)
    )


def eval_df_to_eval_dataset(cfg: PsycopConfig) -> EvalDataset:
    eval_df = pd.read_parquet(f"{cfg['logger']['*']['disk']['run_path']}/eval_df.parquet")

    eval_df["ids"] = eval_df["pred_time_uuid"].apply(lambda s: s.split("-")[0])
    eval_df["pred_timestamps"] = eval_df["pred_time_uuid"].apply(lambda s: s.split("-")[1:4])

    return EvalDataset(
        ids=eval_df["ids"],
        y=eval_df["y"],
        y_hat_probs=eval_df["y_hat_prob"],
        pred_timestamps=eval_df["pred_timestamps"],
        pred_time_uuids=eval_df["pred_time_uuid"],
    )
