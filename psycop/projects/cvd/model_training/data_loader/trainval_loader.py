from pathlib import Path
import polars as pl
from functionalpy import Seq

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader

class MissingPathError(Exception):
    ...


@BaselineRegistry.data.register("parquet_vertical_concatenator")
class ParquetVerticalConcatenator(BaselineDataLoader):
    def __init__(self, *args: str):
        self.dataset_paths = [Path(arg) for arg in args]

        missing_paths = Seq(self.dataset_paths).map(self._check_path_exists).flatten()
        if missing_paths:
            raise MissingPathError("""The following paths are missing:
                {missing_paths}
            """)

    def _check_path_exists(self, path: Path) -> list[MissingPathError]
        if not path.exists():
            return [MissingPathError(path)]

        return []

    def load(self) -> pl.LazyFrame:
        return pl.concat(how="vertical", items=
            [
                pl.scan_parquet(path)
                for path in self.dataset_paths
            ],
        )
