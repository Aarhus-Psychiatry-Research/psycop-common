# Implement this object for cross-validation, split-validation
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import polars as pl

from psycop.common.model_training_v2.trainer.task.base_metric import (
    CalculatedMetric,
)


@dataclass(frozen=True)
class TrainingResult:
    metric: CalculatedMetric
    df: pl.DataFrame


@runtime_checkable
class BaselineTrainer(Protocol):
    def train(self) -> TrainingResult:
        ...

    def _log_sklearn_pipe(self) -> None:
        ...

    def _log_main_metric(self, main_metric: CalculatedMetric) -> None:
        ...
