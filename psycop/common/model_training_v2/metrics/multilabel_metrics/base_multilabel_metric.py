from typing import Protocol

from psycop.common.model_training_v2.metrics.base_metric import CalculatedMetric

from ...presplit_preprocessing.polars_frame import PolarsFrame


class MultilabelMetric(Protocol):
    def calculate(self, y_true: PolarsFrame, y_pred: PolarsFrame) -> CalculatedMetric:
        ...
