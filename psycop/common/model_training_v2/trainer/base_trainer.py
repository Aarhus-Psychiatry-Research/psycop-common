# Implement this object for cross-validation, split-validation

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from psycop.common.model_training_v2.training_method.problem_type.base_metric import CalculatedMetric

from psycop.common.model_training_v2.training_method.problem_type.eval_dataset_base import BaseEvalDataset


@dataclass(frozen=True)
class TrainingResult:
    metric: CalculatedMetric
    eval_dataset: BaseEvalDataset


@runtime_checkable
class Trainer(Protocol):
    def train(self) -> TrainingResult:
        ...
