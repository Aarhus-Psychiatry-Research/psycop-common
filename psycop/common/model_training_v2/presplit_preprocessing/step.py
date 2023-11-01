from typing import Callable, Protocol, TypeVar

import polars as pl

PolarsFrame = pl.LazyFrame | pl.DataFrame
PolarsFrame_T0 = TypeVar("PolarsFrame_T0", bound=PolarsFrame)

class PresplitStep(Protocol):
    def apply(self, input_df: PolarsFrame_T0) -> PolarsFrame_T0:
        ...
