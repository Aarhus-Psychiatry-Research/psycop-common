from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import pandas as pd

    from psycop.common.model_training_v2.training_method.base_training_method import (
        TrainingResult,
    )

    from psycop.common.model_training_v2.training_method.preprocessing.polars_frame import PolarsFrame


class Task(Protocol):
    def train(self, x: PolarsFrame, y: PolarsFrame) -> None:
        """Train the model"""
        ...

    def evaluate(self, x: PolarsFrame, y: PolarsFrame) -> TrainingResult:
        ...

    def predict_proba(self, x: PolarsFrame) -> pd.Series[float]:
        ...
