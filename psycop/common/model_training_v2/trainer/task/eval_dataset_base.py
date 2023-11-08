from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass
class BaseEvalDataset:
    pred_time_uuids: str
    y_hat_probs: str
    y: str
    df: pl.DataFrame

    def to_disk(self, path: Path) -> None:
        ...


