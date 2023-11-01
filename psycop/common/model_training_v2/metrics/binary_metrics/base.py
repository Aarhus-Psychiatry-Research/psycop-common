from typing import Protocol

import polars as pl

from ...presplit_preprocessing.polars_frame import PolarsFrame


class BinaryMetric(Protocol):
    def __call__(self, y_true: PolarsFrame, y_pred: pl.Series) -> float:
        ...
