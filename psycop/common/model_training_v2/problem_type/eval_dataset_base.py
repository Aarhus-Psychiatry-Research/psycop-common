from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from psycop.common.global_utils.pickle import write_to_pickle
from psycop.common.model_training_v2.metrics.base_metric import CalculatedMetric
from psycop.common.model_training_v2.metrics.binary_metrics.base_binary_metric import (
    BinaryMetric,
)


@dataclass
class BaseEvalDataset:
    pred_time_uuids: str
    y_hat_probs: str
    y: str
    df: pl.DataFrame

    def to_disk(self, path: Path) -> None:
        ...


class BinaryEvalDataset(BaseEvalDataset):
    pred_time_uuids: str
    y_hat_probs: str
    y: str
    df: pl.DataFrame

    def calculate_metrics(
        self,
        metrics: Iterable[BinaryMetric],
    ) -> list[CalculatedMetric]:
        return [
            metric.calculate(
                y_true=self.df.get_column(self.y),
                y_pred=self.df.get_column(self.y_hat_probs),
            )
            for metric in metrics
        ]

    def to_disk(self, path: Path) -> None:
        write_to_pickle(self, filepath=path)
