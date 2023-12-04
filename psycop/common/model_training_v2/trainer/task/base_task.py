from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd


@runtime_checkable
class BaselineTask(Protocol):
    def train(self, x: pd.DataFrame, y: pd.DataFrame, y_col_name: str) -> None:
        """Train the model"""
        ...

    def predict_proba(self, x: pd.DataFrame) -> pd.Series[float]:
        ...
