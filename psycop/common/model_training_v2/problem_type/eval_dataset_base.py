from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import polars as pl

from ..presplit_preprocessing.polars_frame import PolarsFrame


@dataclass
class BaseEvalDataset:
    pred_time_uuids: pl.Expr
    y_hat_probs: pl.Expr
    y: pl.Expr
    df: PolarsFrame

    def to_disk(self, path: Path) -> None:
        ...


class BinaryMetric(Protocol):
    def calculate(self, y_hat_probs: Iterable[float], y: Iterable[int]) -> float:
        ...


class BinaryEvalDataset(pl.DataFrame):
    pred_time_uuids: pl.Expr
    y_hat_probs: pl.Expr
    y: pl.Expr
    df: PolarsFrame

    def calculate_metrics(self, metrics: Iterable[BinaryMetric]) -> dict[str, float]:
        ...
