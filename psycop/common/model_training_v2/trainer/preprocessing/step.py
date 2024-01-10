from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class PresplitStep(Protocol):
    def apply(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        ...
