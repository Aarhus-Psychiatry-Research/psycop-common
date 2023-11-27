from pathlib import Path

import polars as pl
import pytest
from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader
from psycop.common.model_training_v2.trainer.data.data_filters.test_data_filter import TestDataFilter

from psycop.common.model_training_v2.trainer.data.dataloaders import (
    DataLoaderFilterer,
    MissingPathError,
    ParquetVerticalConcatenator,
)
from psycop.common.model_training_v2.trainer.data.minimal_test_data import MinimalTestData


def test_vertical_concatenator(tmpdir: Path):
    df = pl.DataFrame({"a": [1, 2, 3]})
    n_paths = 2

    parquet_paths = [Path(tmpdir) / f"test_{i}.parquet" for i in range(n_paths)]

    for p in parquet_paths:
        df.write_parquet(p)

    concatenated = (
        ParquetVerticalConcatenator(
            paths=[str(p) for p in parquet_paths],
        )
        .load()
        .collect()
    )

    assert len(concatenated) == len(df) * n_paths
    assert concatenated.columns == df.columns

    with pytest.raises(MissingPathError):
        ParquetVerticalConcatenator(
            paths=[str(p) for p in parquet_paths] + ["non_existent_path"],
        ).load()


def test_dataloader_filterer():
    base_dataloader = MinimalTestData()
    
    class TestDataFilter:
        def apply(self, dataloader: BaselineDataLoader):
            return dataloader.load().filter(pl.col("dw_ek_borger") == 1)
    
    filter_dataloader = DataLoaderFilterer(dataloader=base_dataloader, filter=TestDataFilter())

    assert filter_dataloader.load().collect().shape == (1, 6)