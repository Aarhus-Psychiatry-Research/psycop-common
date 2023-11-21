from pathlib import Path

import polars as pl

from psycop.common.model_training_v2.trainer.preprocessing.steps.dataloaders import (
    ParquetVerticalConcatenator,
)


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
