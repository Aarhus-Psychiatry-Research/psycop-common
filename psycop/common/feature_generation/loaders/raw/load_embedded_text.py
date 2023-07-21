import polars as pl


# pl scan parquet -> filter by overskrift -> filter by split


from typing import Literal, Sequence


def load_embedded_text(filename: str, text_sfi_names: list[str], include_sfi_name: bool, split_names: Sequence[Literal["train", "val", "test"]], n_rows: int | None) -> pl.DataFrame:
    