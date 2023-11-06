# Implement this object for cross-validation, split-validation

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from psycop.common.model_training_v2.metrics.base_metric import CalculatedMetric
from psycop.common.model_training_v2.problem_type.eval_dataset_base import (
    BaseEvalDataset,
)


@dataclass(frozen=True)
class TrainingResult:
    main_metric: CalculatedMetric
    supplementary_metrics: Sequence[CalculatedMetric] | None
    eval_dataset: BaseEvalDataset


class TrainingMethod(Protocol):
    def train(self) -> TrainingResult:
        ...
