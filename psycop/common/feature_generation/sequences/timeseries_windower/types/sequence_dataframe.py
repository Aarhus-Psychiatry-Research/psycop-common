from dataclasses import dataclass

import polars as pl

from psycop.common.feature_generation.sequences.timeseries_windower.types.abstract_polars_dataframe import (
    ColumnNames,
    PolarsDataframeBundle,
)


@dataclass(frozen=True)
class SequenceColumnNames(ColumnNames):
    entity_id: str = "entity_id"  # E.g. patient ID
    pred_timestamp: str = "pred_timestamp"  # Timestamp for the prediction time
    pred_time_uuid: str = "pred_time_uuid"
    event_source: str = "event_source"  # Values are e.g. "lab"/"diagnosis"/"medication"
    event_type: str = "event_type"  # Values are e.g. "Hba1c"/"Hypertension"/"Metformin"
    event_timestamp: str = "event_timestamp"
    event_value: str = (
        "event_value"  # 1/0 for booleans, numeric value for numeric events
    )


class SequenceDataframeBundle(PolarsDataframeBundle):
    def __init__(
        self,
        df: pl.LazyFrame,
        cols: SequenceColumnNames,
        validate_cols_exist_on_init: bool = True,
    ):
        super().__init__(
            df=df,
            cols=cols,
            validate_cols_exist_on_init=validate_cols_exist_on_init,
        )

        self._cols = cols
        self._frozen = True

    def unpack(self) -> tuple[pl.LazyFrame, SequenceColumnNames]:
        return self._df, self._cols
