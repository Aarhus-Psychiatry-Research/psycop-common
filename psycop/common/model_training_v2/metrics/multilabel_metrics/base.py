from typing import Protocol

from ...presplit_preprocessing.polars_frame import PolarsFrame


class MultilabelMetric(Protocol):
    def __call__(self, y_true: PolarsFrame, y_pred: PolarsFrame) -> float:
        ...
