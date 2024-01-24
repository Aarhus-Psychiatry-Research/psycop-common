from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from ...loggers.supports_logger import SupportsLoggerMixin
from .base_pipeline import BasePipeline

if TYPE_CHECKING:
    import pandas as pd


class BaselineTask(ABC, SupportsLoggerMixin):
    task_pipe: BasePipeline

    def train(self, x: pd.DataFrame, y: pd.DataFrame, y_col_name: str) -> None:
        """Train the model"""
        ...

    def predict_proba(self, x: pd.DataFrame) -> pd.Series[float]:
        ...
