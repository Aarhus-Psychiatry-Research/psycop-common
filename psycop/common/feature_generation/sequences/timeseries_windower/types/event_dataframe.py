from dataclasses import dataclass

import polars as pl

from psycop.common.feature_generation.sequences.timeseries_windower.types.abstract_polars_dataframe import (
    ColumnNames,
    PolarsDataframeBundle,
)


@dataclass(frozen=True)
class EventColumnNames(ColumnNames):
    timestamp: str = "event_timestamp"
    event_source: str = "event_source"  # Values are e.g. "lab"/"diagnosis"/"medication"
    event_type: str = "event_type"  # Values are e.g. "Hba1c"/"Hypertension"/"Metformin"
    event_value: str = (
        "event_value"  # 1/0 for booleans, numeric value for numeric events
    )


class EventDataframeBundle(PolarsDataframeBundle):
    def __init__(
        self,
        df: pl.LazyFrame,
        cols: EventColumnNames = EventColumnNames(),  # noqa: B008
        validate_cols_exist_on_init: bool = True,
    ):
        super().__init__(
            df=df,
            cols=cols,
            validate_cols_exist_on_init=validate_cols_exist_on_init,
        )

        self._cols = cols
        self._frozen = True

    def unpack(self) -> tuple[pl.LazyFrame, EventColumnNames]:
        return self._df, self._cols
