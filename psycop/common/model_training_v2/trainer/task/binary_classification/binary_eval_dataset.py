from collections.abc import Iterable
from pathlib import Path

import polars as pl

from psycop.common.model_training_v2.trainer.task.base_metric import (
    CalculatedMetric,
)
from psycop.common.model_training_v2.trainer.task.binary_classification.binary_metrics import (
    BinaryMetric,
)
from psycop.common.model_training_v2.trainer.task.eval_dataset_base import (
    BaseEvalDataset,
)


class BinaryEvalDataset(BaseEvalDataset):
    pred_time_uuid_col: str
    y_hat_col: str
    y_col: str
    df: pl.DataFrame

    def calculate_metrics(
        self,
        metrics: Iterable[BinaryMetric],
    ) -> list[CalculatedMetric]:
        return [
            metric.calculate(
                y_true=self.df.get_column(self.y_col).to_pandas(),
                y_pred=self.df.get_column(self.y_hat_col).to_pandas(),
            )
            for metric in metrics
        ]

    def to_disk(self, path: Path) -> None:
        self.df.write_parquet(path / "pred_results.parquet")
