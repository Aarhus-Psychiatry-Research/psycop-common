from pathlib import Path

import polars as pl
import pytest
from polars import LazyFrame

from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader
from psycop.common.model_training_v2.trainer.data.dataloaders import (
    MissingPathError,
    ParquetLoader,
    ParquetVerticalConcatenator,
)

from ....test_utils.str_to_df import str_to_pl_df
from ...config.baseline_registry import BaselineRegistry


def test_parquet_loader(tmpdir: Path):
    df = pl.DataFrame({"a": [1, 2, 3]})

    parquet_path = Path(tmpdir) / "test.parquet"

    df.write_parquet(parquet_path)

    parquet = ParquetLoader(str(parquet_path)).load().collect()

    assert len(parquet) == len(df)
    assert parquet.columns == df.columns

    with pytest.raises(MissingPathError):
        ParquetLoader("non_existent_path").load()


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


@BaselineRegistry.data.register("minimal_test_data")
class MinimalTestData(BaselineDataLoader):
    def __init__(self) -> None:
        pass

    def load(self) -> LazyFrame:
        data = str_to_pl_df(
            """ pred_time_uuid, dw_ek_borger, pred_1, outcome,    outcome_val,    pred_age
                1,              1, 1,      1,          1,              1
                2,              2, 1,      1,          1,              99
                3,              3, 1,      1,          1,              99
                4,              4, 0,      0,          0,              99
                5,             5,  0,      0,          0,              99
                6,              6, 0,      0,          0,              99
                                        """
        ).lazy()

        return data
