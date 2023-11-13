from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from psycop.common.model_training_v2.trainer.task.eval_dataset_base import (
    BaseEvalDataset,
)


@dataclass
class CalculatedMetric:
    name: str
    value: float


@runtime_checkable
class BaseMetric(Protocol):
    def calculate(
        self,
        eval_dataset: BaseEvalDataset,
    ) -> CalculatedMetric:
        ...
