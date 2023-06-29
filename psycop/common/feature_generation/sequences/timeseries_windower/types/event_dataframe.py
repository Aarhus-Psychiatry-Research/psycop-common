from dataclasses import dataclass

import polars as pl

from psycop.common.feature_generation.sequences.timeseries_windower.types.abstract_polars_dataframe import (
    ColumnBundle,
    PolarsDataframeBundle,
)


@dataclass(frozen=True)
class EventColumns(ColumnBundle):
    timestamp: str = "event_timestamp"
    event_type: str = "event_type"
    event_source: str = "event_source"
    event_value: str = "event_value"


class EventDataframeBundle(PolarsDataframeBundle):
    def __init__(
        self,
        df: pl.LazyFrame,
        cols: EventColumns,
        validate_cols_exist_on_init: bool = True,
    ):
        super().__init__(
            df=df,
            cols=cols,
            validate_cols_exist_on_init=validate_cols_exist_on_init,
        )

        self._cols = cols
        self._frozen = True

    def unpack(self) -> tuple[pl.LazyFrame, EventColumns]:
        return self._df, self._cols
