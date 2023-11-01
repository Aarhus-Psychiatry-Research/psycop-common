from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import polars as pl


@dataclass
class BaseEvalDataset(pl.DataFrame):
    pred_time_uuids: pl.Expr = field(default=pl.col("pred_time_uuids"))
    y_hat_probs: pl.Expr = field(default=pl.col("y_hat_probs"))
    y: pl.Expr = field(default=pl.col("y"))

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
