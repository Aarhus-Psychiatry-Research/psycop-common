from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ...loggers.supports_logger import SupportsLoggerMixin

if TYPE_CHECKING:
    import pandas as pd

    from .base_pipeline import BasePipeline


class BaselineTask(ABC, SupportsLoggerMixin):
    task_pipe: BasePipeline

    @abstractmethod
    def train(self, x: pd.DataFrame, y: pd.DataFrame, y_col_name: str) -> None:
        """Train the model"""
        ...

    @abstractmethod
    def predict_proba(self, x: pd.DataFrame) -> pd.Series[float]:
        ...
