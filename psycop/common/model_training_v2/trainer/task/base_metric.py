from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from psycop.common.model_training_v2.trainer.task.binary_classification.binary_classification_pipeline import (
    PredProbaSeries,
)


@dataclass
class CalculatedMetric:
    name: str
    value: float


class BaseMetric(Protocol):
    def calculate(
        self,
        y_true: pd.Series[int],
        y_pred: PredProbaSeries,
    ) -> CalculatedMetric:
        ...
