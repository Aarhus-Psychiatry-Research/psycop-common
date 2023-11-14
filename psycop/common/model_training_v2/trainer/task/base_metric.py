from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd


@dataclass
class CalculatedMetric:
    name: str
    value: float


PredProbaSeries = pd.Series  # name should be "y_hat_probs", series of floats

@runtime_checkable
class BaseMetric(Protocol):
    def calculate(
        self,
        y: pd.Series,  # type: ignore
        y_hat_prob: PredProbaSeries,
    ) -> CalculatedMetric:
        ...
