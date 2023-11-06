from typing import Protocol

import polars as pl

from psycop.common.model_training_v2.metrics.base_metric import CalculatedMetric


class BinaryMetric(Protocol):
    def calculate(self, y_true: pl.Series, y_pred: pl.Series) -> CalculatedMetric:
        ...
