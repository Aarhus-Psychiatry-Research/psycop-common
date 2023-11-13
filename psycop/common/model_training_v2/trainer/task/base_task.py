from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd

    from psycop.common.model_training_v2.trainer.base_trainer import TrainingResult


@runtime_checkable
class BaselineTask(Protocol):
    def train(self, x: pd.DataFrame, y: pd.DataFrame, y_col_name: str) -> None:
        """Train the model"""
        ...

    def evaluate(
        self,
        df: pd.DataFrame,
        y_hat_col: str,
        y_col: str,
    ) -> TrainingResult:
        ...

    def predict_proba(self, x: pd.DataFrame) -> pd.Series[float]:
        ...
