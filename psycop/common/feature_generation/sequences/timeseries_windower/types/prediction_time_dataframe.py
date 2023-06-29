from dataclasses import asdict, dataclass

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
class PredictionTimeColumns(ColumnBundle):
    timestamp: str = "pred_timestamp"
    pred_time_uuid: str = "pred_time_uuid"


@dataclass(frozen=True)
class PredictiontimeDataframeBundle(PolarsDataframeBundle):
    _df: pl.LazyFrame
    _cols: PredictionTimeColumns = PredictionTimeColumns()  # noqa: RUF009

    def unpack(self) -> tuple[pl.LazyFrame, PredictionTimeColumns]:
        if self._cols.pred_time_uuid not in self._df.columns:
            df = self._df.with_columns(
                create_pred_time_uuids(
                    entity_id_col_name=self._cols.entity_id,
                    timestamp_col_name=self._cols.timestamp,
                ).alias(self._cols.pred_time_uuid),
            )
        else:
            df = self._df

        for _, col_name in asdict(self._cols).items():
            if col_name not in df.columns:
                raise pl.ColumnNotFoundError(
                    f"Column {col_name} not found in dataframe",
                )

        return (
            df,
            self._cols,
        )
