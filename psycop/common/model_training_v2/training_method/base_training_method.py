# Implement this object for cross-validation, split-validation

from dataclasses import dataclass
from typing import Protocol

from psycop.common.model_training_v2.metrics.base_metric import CalculatedMetric

from ..problem_type.eval_dataset_base import BaseEvalDataset


@dataclass(frozen=True)
class TrainingResult:
    metric: CalculatedMetric
    eval_dataset: BaseEvalDataset


class TrainingMethod(Protocol):
    def train(self) -> TrainingResult:
        ...
