from pathlib import Path

import polars as pl

from psycop.common.model_training_v2.trainer.task.eval_dataset_base import (
    BaseEvalDataset,
)


class BinaryEvalDataset(BaseEvalDataset):
    pred_time_uuid_col: str
    y_hat_col: str
    y_col: str
    df: pl.DataFrame

    def to_disk(self, path: Path) -> None:
        self.df.write_parquet(path / "pred_results.parquet")
