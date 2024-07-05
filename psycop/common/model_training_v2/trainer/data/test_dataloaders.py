from dataclasses import dataclass
from logging.config import dictConfig
from pathlib import Path

import polars as pl
import pytest
from polars import LazyFrame

from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader
from psycop.common.model_training_v2.trainer.data.dataloaders import (
    MissingPathError,
    ParquetVerticalConcatenator,
)

from ....test_utils.str_to_df import str_to_pl_df
from ...config.baseline_registry import BaselineRegistry


def test_vertical_concatenator(tmpdir: Path):
    df = pl.DataFrame({"a": [1, 2, 3]})
    n_paths = 2

    parquet_paths = [Path(tmpdir) / f"test_{i}.parquet" for i in range(n_paths)]

    for p in parquet_paths:
        df.write_parquet(p)

    concatenated = (
        ParquetVerticalConcatenator(paths=[str(p) for p in parquet_paths]).load().collect()
    )

    assert len(concatenated) == len(df) * n_paths
    assert concatenated.columns == df.columns

    with pytest.raises(MissingPathError):
        ParquetVerticalConcatenator(
            paths=[str(p) for p in parquet_paths] + ["non_existent_path"]
        ).load()


@dataclass
class MinimalTestRow:
    pred_time_uuid: int
    dw_ek_borger: int
    pred_1: int
    outcome: int
    outcome_val: int
    pred_age: int

    def to_str(self) -> str:
        return f"{self.pred_time_uuid}, {self.dw_ek_borger}, {self.pred_1}, {self.outcome}, {self.outcome_val}, {self.pred_age}"

    @classmethod
    def col_str(cls: type["MinimalTestRow"]) -> str:
        return "pred_time_uuid, dw_ek_borger, pred_1, outcome, outcome_val, pred_age"


@BaselineRegistry.data.register("minimal_test_data")
class MinimalTestData(BaselineDataLoader):
    def __init__(self, n: int = 6) -> None:
        self.n = n

    def load(self) -> LazyFrame:
        rows = [MinimalTestRow.col_str()]
        for i in range(self.n):
            outcome = 1 if i < self.n // 2 else 0
            rows.append(
                MinimalTestRow(
                    pred_time_uuid=i + 1,
                    dw_ek_borger=i + 1,
                    pred_1=1,
                    outcome=outcome,
                    outcome_val=outcome,
                    pred_age=1 if i == 0 else 99,
                ).to_str()
            )

        data_str = "\n".join(rows)
        return str_to_pl_df(data_str).lazy()
