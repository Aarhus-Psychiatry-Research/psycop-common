from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class BaselineDataLoader(Protocol):
    def load(self) -> pl.LazyFrame:
        ...