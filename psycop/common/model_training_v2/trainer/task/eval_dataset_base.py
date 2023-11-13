from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from psycop.common.model_training_v2.trainer.task.base_metric import (
    BaseMetric,
    CalculatedMetric,
)


@dataclass
class BaseEvalDataset:
    pred_time_uuid_col: str
    y_hat_col: str
    y_col: str
    df: pl.DataFrame

    def calculate_metrics(
        self,
        metrics: Iterable[BaseMetric],
    ) -> list[CalculatedMetric]:
        ...

    def to_disk(self, path: Path) -> None:
        ...
