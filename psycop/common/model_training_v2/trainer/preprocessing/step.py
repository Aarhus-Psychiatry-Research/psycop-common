from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import polars as pl

from ...loggers.base_logger import BaselineLogger
from ...loggers.supports_logger import SupportsLoggerMixin


class PresplitStep(ABC, SupportsLoggerMixin):
    @abstractmethod
    def apply(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        ...
