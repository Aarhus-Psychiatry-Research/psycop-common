import json
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
import polars as pl
from psycop.common.model_training.config_schemas.conf_utils import (
    FullConfigSchema,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from sklearn.pipeline import Pipeline


@dataclass
class RunGroup:
    name: str

    @property
    def group_dir(self) -> Path:
        return Path(f"E:/shared_resources/coercion/model_eval/{self.name}")

    @property
    def flattened_ds_dir(self) -> Path:
        first_run = list(self.group_dir.glob(r"*"))[0]

        config_path = first_run / "cfg.json"

        with config_path.open() as f:
            config_str = json.load(f)
            config_dict = json.loads(config_str)

        return Path(config_dict["data"]["dir"])

    @property
    def all_runs_performance_df(self) -> pd.DataFrame:
        run_performance_files = self.group_dir.glob("*.parquet")

        concatenated_performance_df = pd.concat(
            pd.read_parquet(parquet_file) for parquet_file in run_performance_files
        )

        return concatenated_performance_df

    def get_best_runs_by_lookahead(self) -> pl.DataFrame:
        df = pl.from_pandas(self.all_runs_performance_df)

        return (
            df.groupby(["lookahead_days", "model_name"])
            .agg(pl.all().sort_by("roc_auc", descending=True).first())
            .sort(["model_name", "lookahead_days"])
        )


SplitNames = Literal["train", "test", "val"]


@dataclass
class Run:
    group: RunGroup
    name: str
    pos_rate: float

    def _get_flattened_split_path(self, split: SplitNames) -> Path:
        matches = list(self.group.flattened_ds_dir.glob(f"*{split}*.parquet"))

        if len(matches) != 1:
            raise ValueError("More than one matching split file found")
        return matches[0]

    def get_flattened_split_as_pd(self, split: SplitNames) -> pd.DataFrame:
        return pd.read_parquet(self._get_flattened_split_path(split=split))

    def get_flattened_split_as_lazyframe(self, split: SplitNames) -> pl.LazyFrame:
        return pl.scan_parquet(self._get_flattened_split_path(split=split))

    @property
    def cfg(self) -> FullConfigSchema:
        # Loading the json instead of the .pkl makes us independent
        # of whether the imports in psycop-common model-training have changed
        return FullConfigSchema.parse_obj(self.get_cfg_as_json())

    @property
    def eval_dir(self) -> Path:
        return self.group.group_dir / self.name

    @property
    def model_type(self) -> str:
        return self.cfg.model.name

    def get_cfg_as_json(self) -> FullConfigSchema:
        # Load json
        path = self.eval_dir / "cfg.json"
        return json.loads(json.loads(path.read_text()))

    def get_eval_dataset(
        self,
        custom_columns: Optional[Sequence[str]] = None,
    ) -> EvalDataset:
        df = pd.read_parquet(self.eval_dir / "evaluation_dataset.parquet")

        eval_dataset = df_to_eval_dataset(df, custom_columns=custom_columns)

        return eval_dataset

    def get_auroc(self) -> float:
        df = self.group.all_runs_performance_df

        self_run = df[df["run_name"] == self.name]

        return self_run["roc_auc"].iloc[0]

    @property
    def pipe(self) -> Pipeline:
        return load_file_from_pkl(self.eval_dir / "pipe.pkl")


def load_file_from_pkl(file_path: Path) -> Any:
    with file_path.open("rb") as f:
        return pickle.load(f)


def df_to_eval_dataset(
    df: pd.DataFrame,
    custom_columns: Optional[Sequence[str]],
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
        pred_time_uuids=df["pred_time_uuids"],
        custom_columns={col: df[col] for col in custom_columns}
        if custom_columns
        else None,
    )
