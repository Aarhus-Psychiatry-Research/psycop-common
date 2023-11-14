from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from psycop.common.model_training_v2.trainer.task.binary_classification.binary_classification_pipeline import (
    PredProbaSeries,
)


@dataclass
class CalculatedMetric:
    name: str
    value: float


@runtime_checkable
class BaseMetric(Protocol):
    def calculate(
        self,
        y: pd.Series,  # type: ignore
        y_hat_prob: PredProbaSeries,
    ) -> CalculatedMetric:
        ...
