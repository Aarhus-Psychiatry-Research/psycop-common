from dataclasses import dataclass
from typing import Sequence

import polars as pl

from psycop.common.feature_generation.sequences.timeseries_windower.types.abstract_polars_dataframe import (
    ColumnBundle,
    PolarsDataframeBundle,
)


@dataclass(frozen=True)
class SequenceDataframeColumns(ColumnBundle):
    entity_id: str = "entity_id"
    pred_timestamp: str = "pred_timestamp"
    pred_time_uuid: str = "pred_time_uuid"
    event_type: str = "event_type"
    event_timestamp: str = "event_timestamp"
    event_source: str = "event_source"
    event_value: str = "event_value"


@dataclass(frozen=True)
class SequenceDataframeBundle(PolarsDataframeBundle):
    _df: pl.LazyFrame
    _cols: SequenceDataframeColumns

    def unpack(self) -> tuple[pl.LazyFrame, SequenceDataframeColumns]:
        self._validate_col_names()
        return self._df, self._cols
