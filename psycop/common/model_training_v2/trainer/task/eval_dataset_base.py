from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass
class BaseEvalDataset:
    pred_time_uuid_col: str
    y_hat_col: str
    y_col: str
    df: pl.DataFrame

    def to_disk(self, path: Path) -> None:
        ...
