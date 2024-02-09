from collections.abc import Sequence
from pathlib import Path

import polars as pl
from iterpy import Iter

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader


class MissingPathError(Exception):
    ...


@BaselineRegistry.data.register("parquet_loader")
class ParquetLoader(BaselineDataLoader):
    def __init__(self, path: str, validate_on_init: bool = True) -> None:
        """Load single parquet file.

        Args:
            path: Path to parquet file.
            validate_on_init: Whether to validate the path on init.
                Helpful when testing the .cfg parses, where the absolute path will differ between devcontainer and Ovartaci.
                Defaults to True.

        """
        self.dataset_paths = Path(path)

        if validate_on_init:
            missing_path = self._check_path_exists(self.dataset_paths)
            if missing_path:
                raise MissingPathError(
                    f"""The following paths are missing:
                    {missing_path}
                """
                )

    def _check_path_exists(self, path: Path) -> list[MissingPathError]:
        if not path.exists():
            return [MissingPathError(path)]

        return []

    def load(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.dataset_paths)


@BaselineRegistry.data.register("parquet_vertical_concatenator")
class ParquetVerticalConcatenator(BaselineDataLoader):
    def __init__(self, paths: Sequence[str], validate_on_init: bool = True):
        """Vertical concatenation of multiple parquet files.

        Args:
            paths: Paths to parquet files.
            logger: Logger to use.
            validate_on_init: Whether to validate the paths on init.
                Helpful when testing the .cfg parses, where the absolute path will differ between devcontainer and Ovartaci.
                Defaults to True.

        """
        self.dataset_paths = [Path(arg) for arg in paths]

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
        return pl.concat(
            how="vertical", items=[pl.scan_parquet(path) for path in self.dataset_paths]
        )


if __name__ == "__main__":
    concatenator = ParquetVerticalConcatenator(
        ["/home/psycop/psycop/data/processed/2021-09-01/2021-09-01.parquet"], validate_on_init=False
    )
