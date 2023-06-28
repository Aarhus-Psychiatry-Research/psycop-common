from dataclasses import asdict, dataclass

import polars as pl


@dataclass(frozen=True)
class ColumnBundle:
    entity_id: str = "entity_id"


@dataclass(frozen=True)
class PolarsDataframeBundle:
    _df: pl.LazyFrame
    _cols: ColumnBundle

    def _validate_col_names(self):
        # Check col_names exist
        for _, col_name in asdict(self._cols).items():
            if col_name not in self._df.columns:
                raise pl.ColumnNotFoundError(
                    f"Column {col_name} found in _cols but not in dataframe, columns in df: {self._df.columns}",
                )
