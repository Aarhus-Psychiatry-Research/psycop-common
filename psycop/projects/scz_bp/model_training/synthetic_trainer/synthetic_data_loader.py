from collections.abc import Sequence
from pathlib import Path

import polars as pl
from iterpy import Iter

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader


class MissingPathError(Exception): ...


@BaselineRegistry.data.register("synthetic_data_vertical_concatenator")
class SyntheticVerticalConcatenator(BaselineDataLoader):
    def __init__(self, paths: Sequence[str], n_samples: int | None, validate_on_init: bool = True):
        """Vertical concatenation of multiple parquet files with option for
        choosing only top n_samples
        """
        self.dataset_paths = [Path(arg) for arg in paths]
        self.n_samples = n_samples

        if validate_on_init:
            missing_paths = (
                Iter(self.dataset_paths).map(self._check_path_exists).flatten().to_list()
            )
            if missing_paths:
                raise MissingPathError(
                    f"""The following paths are missing:
                    {missing_paths}
                """
                )

    def _check_path_exists(self, path: Path) -> list[MissingPathError]:
        if not path.exists():
            return [MissingPathError(path)]

        return []

    def load(self) -> pl.LazyFrame:
        df = pl.concat(how="vertical", items=[pl.scan_parquet(path) for path in self.dataset_paths])
        if self.n_samples is not None:
            return df.head(n=self.n_samples)
        return df
