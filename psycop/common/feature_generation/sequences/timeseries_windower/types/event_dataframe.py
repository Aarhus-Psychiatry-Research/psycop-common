from dataclasses import dataclass

import polars as pl

from psycop.common.feature_generation.sequences.timeseries_windower.types.abstract_polars_dataframe import (
    ColumnBundle,
    PolarsDataframeBundle,
)


@dataclass(frozen=True)
class EventDataframeColumns(ColumnBundle):
    timestamp: str = "event_timestamp"
    event_type: str = "event_type"
    event_source: str = "event_source"
    event_value: str = "event_value"


@dataclass(frozen=True)
class EventDataframeBundle(PolarsDataframeBundle):
    _df: pl.LazyFrame
    _cols: EventDataframeColumns = EventDataframeColumns()  # noqa: RUF009

    def unpack(self) -> tuple[pl.LazyFrame, EventDataframeColumns]:
        self._validate_col_names()
        return self._df, self._cols
