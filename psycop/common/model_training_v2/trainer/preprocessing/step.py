from abc import ABC, abstractmethod
from dataclasses import dataclass

import polars as pl

from ...loggers.supports_logger import SupportsLoggerMixin


class PresplitStep(ABC, SupportsLoggerMixin):
    @abstractmethod
    def apply(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        ...
