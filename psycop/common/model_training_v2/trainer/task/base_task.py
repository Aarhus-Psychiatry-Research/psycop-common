from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import pandas as pd

    from psycop.common.model_training_v2.trainer.base_trainer import TrainingResult
    from psycop.common.model_training_v2.trainer.preprocessing.polars_frame import (
        PolarsFrame,
    )

class BaselineTask(Protocol):
    def train(self, x: PolarsFrame, y: PolarsFrame) -> None:
        """Train the model"""
        ...

    def evaluate(self, x: PolarsFrame, y: PolarsFrame) -> TrainingResult:
        ...

    def predict_proba(self, x: PolarsFrame) -> pd.Series[float]:
        ...
