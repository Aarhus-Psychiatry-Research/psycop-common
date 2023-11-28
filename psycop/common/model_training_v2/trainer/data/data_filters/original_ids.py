from collections.abc import Collection
from typing import Literal

import polars as pl

from psycop.common.feature_generation.loaders.raw.load_ids import SplitName, load_ids
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader


@BaselineRegistry.data_filters.register("original_id_data_filter")
class OriginalIDDataFilter:
    def __init__(
        self,
        splits_to_keep: Collection[Literal["train", "val", "test"]] | None,
        split_series: pl.Series | None = None,
        id_col_name: str = "dw_ek_borger",
    ):
        """Filter the data to only include ids from the original datasplit.
        If `split_series` is None, will load the ids from the original datasplit
        based on the `splits_to_keep` argument. If `split_df` is not None,
        provide a pl.Series with the ids to keep."""
        if splits_to_keep is not None and split_series is None:
            raise ValueError(
                """splits_to_keep must be supplied when using the original
                id split.""",
            )
        if splits_to_keep is not None and split_series is not None:
            raise ValueError(
                """splits_to_keep and split_series cannot both be supplied.
                splits_to_keep is used to load the ids from the original
                datasplit, while split_series is used to filter the data
                based on a custom id split.""",
            )

        self.splits_to_keep = splits_to_keep
        self.id_col_name = id_col_name
        if split_series is None:
            self.split_series = self._load_and_prepare_original_ids_df()
        else:
            self.split_series = split_series

    def apply(self, dataloader: BaselineDataLoader) -> pl.LazyFrame:
        """Filter the dataloader to only include ids from the desired splits
        from the original datasplit"""
        return dataloader.load().filter(
            pl.col(self.id_col_name).is_in(self.split_series)
        )

    def _load_and_prepare_original_ids_df(self) -> pl.Series:
        split_names: list[SplitName] = []

        for split_name in self.splits_to_keep:  # type: ignore already guarded in init
            match split_name:
                case "train":
                    split_names.append(SplitName.TRAIN)
                case "val":
                    split_names.append(SplitName.VALIDATION)
                case "test":
                    split_names.append(SplitName.TEST)
                case _:
                    raise ValueError(
                        f"Splitname {split_name} is not allowed, try from ['train', 'test', 'val']",
                    )
        return pl.concat(
            [pl.from_pandas(load_ids(split)) for split in split_names],
        ).get_column("dw_ek_borger")
