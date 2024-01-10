from typing import Protocol, TypeVar, runtime_checkable

import polars as pl

from psycop.common.types.polarsframe import (
    PolarsFrame,
)


@runtime_checkable
class PresplitStep(Protocol):
    def apply(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        ...
