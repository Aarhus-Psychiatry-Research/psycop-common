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
        for _, col_name in asdict(self._cols).items():
            missing_cols = []

            if col_name not in self._df.columns:
                missing_cols.append(col_name)

            if len(missing_cols) > 0:
                cols_attr_string = f"{self.__class__.__name__}._cols"
                df_attr_string = f"{self.__class__.__name__}._df"
                raise pl.ColumnNotFoundError(
                    f"""Column(s) {missing_cols} required by {cols_attr_string} but missing in {df_attr_string}.
    Columns in {df_attr_string}: {self._df.columns}
    Columns in {cols_attr_string}: {[c for _, c in self._cols.__dict__.items()]}
""",
                )

    def __setattr__(self, name: str, value: Any) -> None:
        # Check if self._frozen exists and is True
        if getattr(self, "_frozen", False):
            raise AttributeError(
                f"Trying to set attribute on a frozen instance of {self.__class__.__name__}",
            )
        return super().__setattr__(name, value)
