from dataclasses import dataclass

import polars as pl

from psycop.common.feature_generation.sequences.timeseries_windower.types.abstract_polars_dataframe import (
    ColumnBundle,
    PolarsDataframeBundle,
)


@dataclass(frozen=True)
class SequenceColumns(ColumnBundle):
    entity_id: str = "entity_id"
    pred_timestamp: str = "pred_timestamp"
    pred_time_uuid: str = "pred_time_uuid"
    event_type: str = "event_type"
    event_timestamp: str = "event_timestamp"
    event_source: str = "event_source"
    event_value: str = "event_value"


class SequenceDataframeBundle(PolarsDataframeBundle):
    def __init__(
        self,
        df: pl.LazyFrame,
        cols: SequenceColumns,
        validate_cols_exist_on_init: bool = True,
    ):
        super().__init__(
            df=df,
            cols=cols,
            validate_cols_exist_on_init=validate_cols_exist_on_init,
        )

        self._cols = cols
        self._frozen = True

    def unpack(self) -> tuple[pl.LazyFrame, SequenceColumns]:
        return self._df, self._cols
