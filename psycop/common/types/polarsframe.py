from typing import TypeVar

import polars as pl

PolarsFrame = pl.LazyFrame | pl.DataFrame

PolarsFrameGeneric = TypeVar("PolarsFrameGeneric", bound=PolarsFrame)
