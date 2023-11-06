from typing import Protocol

import polars as pl

from psycop.common.model_training_v2.training_method.base_training_method import (
    TrainingResult,
)

from ..presplit_preprocessing.polars_frame import PolarsFrame


class ProblemType(Protocol):
    def train(self, x: PolarsFrame, y: PolarsFrame) -> None:
        """Train the model"""
        ...

    def evaluate(self, x: PolarsFrame, y: PolarsFrame) -> TrainingResult:
        ...

    def predict_proba(self, x: PolarsFrame) -> pl.Series:
        ...
