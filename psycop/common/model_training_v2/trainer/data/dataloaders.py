from collections.abc import Sequence
from pathlib import Path

import polars as pl
from functionalpy import Seq

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader
from psycop.common.model_training_v2.trainer.data.data_filters.base_data_filter import (
    BaselineDataFilter,
)


class MissingPathError(Exception):
    ...


@BaselineRegistry.data.register("parquet_vertical_concatenator")
class ParquetVerticalConcatenator(BaselineDataLoader):
    def __init__(self, paths: Sequence[str], validate_on_init: bool = True):
        """Vertical concatenation of multiple parquet files.

        Args:
            paths: Paths to parquet files.
            validate_on_init: Whether to validate the paths on init.
                Helpful when testing the .cfg parses, where the absolute path will differ between devcontainer and Ovartaci.
                Defaults to True.

        """
        self.dataset_paths = [Path(arg) for arg in paths]

        if validate_on_init:
            missing_paths = (
                Seq(self.dataset_paths).map(self._check_path_exists).flatten().to_list()
            )
            if missing_paths:
                raise MissingPathError(
                    f"""The following paths are missing:
                    {missing_paths}
                """,
                )

    def _check_path_exists(self, path: Path) -> list[MissingPathError]:
        if not path.exists():
            return [MissingPathError(path)]

        return []

    def load(self) -> pl.LazyFrame:
        return pl.concat(
            how="vertical",
            items=[pl.scan_parquet(path) for path in self.dataset_paths],
        )


@BaselineRegistry.data.register("filtered_dataloader")
class FilteredDataLoader(BaselineDataLoader):
    """Filter the rows from dataloader using a filter, such as by geographical region,
    id split, or other filters."""

    def __init__(self, dataloader: BaselineDataLoader, data_filter: BaselineDataFilter):
        self.dataloader = dataloader
        self.data_filter = data_filter

    def load(self) -> pl.LazyFrame:
        return self.data_filter.apply(self.dataloader)
