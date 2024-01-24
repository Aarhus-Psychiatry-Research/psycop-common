from abc import ABC, abstractmethod

import polars as pl

from ..loggers.supports_logger import SupportsLoggerMixin


class BaselineDataLoader(ABC, SupportsLoggerMixin):
    @abstractmethod
    def load(self) -> pl.LazyFrame:
        ...
