from dataclasses import asdict, dataclass
from typing import Any

import polars as pl


@dataclass(frozen=True)
class ColumnBundle:
    entity_id: str = "entity_id"


class PolarsDataframeBundle:
    def __init__(
        self,
        df: pl.LazyFrame,
        cols: ColumnBundle,
        validate_cols_exist_on_init: bool = True,
    ):
        self._df = df
        self._cols = cols

        if validate_cols_exist_on_init:
            self._validate_cols_exist()

    def _validate_cols_exist(self):
        # Check col_names exist
        bundle_cols = set(asdict(self._cols).values())
        df_cols = set(self._df.columns)
        missing_cols = bundle_cols - df_cols

        if len(missing_cols) > 0:
            cols_attr_string = f"{self.__class__.__name__}._cols"
            df_attr_string = f"{self.__class__.__name__}._df"
            raise pl.ColumnNotFoundError(
                f"""Column(s) {missing_cols} required by {cols_attr_string} but missing in {df_attr_string}.
Columns required by {cols_attr_string}: {bundle_cols}
Columns in {df_attr_string}: {df_cols}
""",
            )

    def __setattr__(self, name: str, value: Any) -> None:
        # Check if self._frozen exists and is True
        if getattr(self, "_frozen", False):
            raise AttributeError(
                f"Trying to set attribute on a frozen instance of {self.__class__.__name__}",
            )
        return super().__setattr__(name, value)
