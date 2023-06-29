from dataclasses import dataclass

import polars as pl

from psycop.common.feature_generation.sequences.timeseries_windower.types.abstract_polars_dataframe import (
    ColumnBundle,
    PolarsDataframeBundle,
)


def create_pred_time_uuids(entity_id_col_name: str, timestamp_col_name: str) -> pl.Expr:
    return pl.concat_str(
        pl.col(entity_id_col_name),
        pl.lit("_"),
        pl.col(timestamp_col_name).dt.strftime("%y-%m-%d_%H-%M-%S"),
    )


@dataclass(frozen=True)
class PredictiontimeColumns(ColumnBundle):
    timestamp: str = "pred_timestamp"
    pred_time_uuid: str = "pred_time_uuid"


class PredictiontimeDataframeBundle(PolarsDataframeBundle):
    def __init__(
        self,
        df: pl.LazyFrame,
        cols: PredictiontimeColumns = PredictiontimeColumns(),  # noqa: B008
        validate_cols_exist_on_init: bool = True,
    ):
        self._df = df
        self._cols = cols

        if self._cols.pred_time_uuid not in self._df.columns:
            self._df = self._df.with_columns(
                create_pred_time_uuids(
                    entity_id_col_name=self._cols.entity_id,
                    timestamp_col_name=self._cols.timestamp,
                ).alias(self._cols.pred_time_uuid),
            )
        else:
            df = self._df

        if validate_cols_exist_on_init:
            self._validate_cols_exist()

        self._frozen = True

    def unpack(self) -> tuple[pl.LazyFrame, PredictiontimeColumns]:
        return self._df, self._cols
