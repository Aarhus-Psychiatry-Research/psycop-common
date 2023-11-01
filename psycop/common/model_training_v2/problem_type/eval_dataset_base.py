from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import polars as pl


@dataclass
class BaseEvalDataset(pl.DataFrame):
    pred_time_uuids: pl.col
    y_hat_probs: pl.col
    y: pl.col

    def to_disk(self, path: Path) -> None:
        ...


class BinaryMetric(Protocol):
    def calculate(self, y_hat_probs: Iterable[float], y: Iterable[int]) -> float:
        ...


class BinaryEvalDataset(pl.DataFrame):
    pred_time_uuids: pl.col
    y_hat_probs: pl.col
    y: pl.col

    def calculate_metrics(self, metrics: Iterable[BinaryMetric]) -> dict[str, float]:
        ...
