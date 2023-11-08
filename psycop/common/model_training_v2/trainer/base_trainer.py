# Implement this object for cross-validation, split-validation

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from psycop.common.model_training_v2.trainer.task.base_metric import (
    CalculatedMetric,
)
from psycop.common.model_training_v2.trainer.task.eval_dataset_base import (
    BaseEvalDataset,
)


@dataclass(frozen=True)
class TrainingResult:
    metric: CalculatedMetric
    eval_dataset: BaseEvalDataset


@runtime_checkable
class BaselineTrainer(Protocol):
    def train(self) -> TrainingResult:
        ...
