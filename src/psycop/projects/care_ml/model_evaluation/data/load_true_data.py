import pickle
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from joblib import Memory
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from sklearn.pipeline import Pipeline

# create a memory cache with a directory to store the cache
memory = Memory(
    location=Path("E:/shared_resources/coercion/model_eval/model_eval_cache"),
)


def df_to_eval_dataset(
    df: pd.DataFrame,
    custom_columns: Optional[list[str]],
) -> EvalDataset:
    """Convert dataframe to EvalDataset."""
    return EvalDataset(
        ids=df["ids"],
        y=df["y"],
        y_hat_probs=df["y_hat_probs"],
        pred_timestamps=df["pred_timestamps"],
        outcome_timestamps=df["outcome_timestamps"],
        age=df["age"],
        is_female=df["is_female"],
        pred_time_uuids=df["adm_id"],
        custom_columns={col: df[col] for col in custom_columns}
        if custom_columns
        else None,
    )


def get_run_item_file_path(wandb_group: str, wandb_run: str, file_name: str) -> Path:
    return Path(
        f"E:/shared_resources/coercion/model_eval/{wandb_group}/{wandb_run}/{file_name}",
    )


@memory.cache()
def load_eval_dataset(
    wandb_group: str,
    wandb_run: str,
    custom_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    path = get_run_item_file_path(
        wandb_group=wandb_group,
        wandb_run=wandb_run,
        file_name="evaluation_dataset.parquet",
    )
    df = pd.read_parquet(path)

    df_to_eval_dataset(df, custom_columns=custom_columns)

    return df


def load_eval_df(
    wandb_group: str,
    wandb_run: str,
) -> pd.DataFrame:
    path = get_run_item_file_path(
        wandb_group=wandb_group,
        wandb_run=wandb_run,
        file_name="evaluation_dataset.parquet",
    )

    return pd.read_parquet(path)


def load_file_from_pkl(wandb_group: str, wandb_run: str, file_name: str) -> Any:
    path = get_run_item_file_path(
        wandb_group=wandb_group,
        wandb_run=wandb_run,
        file_name=file_name,
    )

    with path.open("rb") as f:
        return pickle.load(f)


def load_fullconfig(wandb_group: str, wandb_run: str) -> FullConfigSchema:
    return load_file_from_pkl(
        wandb_group=wandb_group,
        wandb_run=wandb_run,
        file_name="cfg.pkl",
    )


def load_pipe(wandb_group: str, wandb_run: str) -> Pipeline:
    return load_file_from_pkl(
        wandb_group=wandb_group,
        wandb_run=wandb_run,
        file_name="pipe.pkl",
    )
