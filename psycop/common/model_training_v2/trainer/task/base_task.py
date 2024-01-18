from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .base_pipeline import BasePipeline

if TYPE_CHECKING:
    import pandas as pd


@runtime_checkable
class BaselineTask(Protocol):
    pipe: BasePipeline

    def train(self, x: pd.DataFrame, y: pd.DataFrame, y_col_name: str) -> None:
        """Train the model"""
        ...

    def predict_proba(self, x: pd.DataFrame) -> pd.Series[float]:
        ...
