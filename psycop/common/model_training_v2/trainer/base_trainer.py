# Implement this object for cross-validation, split-validation
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import polars as pl

from psycop.common.model_training_v2.trainer.task.base_metric import (
    CalculatedMetric,
)
from psycop.common.model_training_v2.trainer.task.base_task import BaselineTask


@dataclass(frozen=True)
class TrainingResult:
    metric: CalculatedMetric
    df: pl.DataFrame


@runtime_checkable
class BaselineTrainer(Protocol):
    task: BaselineTask

    def train(self) -> TrainingResult:
        ...
