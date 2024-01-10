from collections.abc import Sequence
from typing import Literal

import polars as pl

from psycop.common.feature_generation.loaders.raw.load_ids import (
    SplitName,
    load_stratified_by_outcome_split_ids,
)
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader
from psycop.common.model_training_v2.trainer.preprocessing.step import PresplitStep


@BaselineRegistry.preprocessing.register("id_data_filter")
class IDDataFilter(PresplitStep):
    def __init__(
        self,
        splits_to_keep: Sequence[Literal["train", "val", "test"]] | None,
        split_ids: Sequence[int] | pl.Series | None = None,
        id_col_name: str = "dw_ek_borger",
    ):
        """Filter the data to only include ids from the original datasplit or a
        custom data split.
        If `split_series` is None, will load the ids from the original datasplit
        based on the `splits_to_keep` argument. If `split_ids` is not None,
        provide a sequence with the ids to keep."""
        if splits_to_keep is None and split_ids is None:
            raise ValueError(
                """splits_to_keep must be supplied when using the original
                id split.""",
            )
        if splits_to_keep is not None and split_ids is not None:
            raise ValueError(
                """splits_to_keep and split_ids cannot both be supplied.
                splits_to_keep is used to load the ids from the original
                datasplit, while split_ids is used to filter the data
                based on a custom id split.""",
            )

        self.splits_to_keep = splits_to_keep
        self.id_col_name = id_col_name
        if split_ids is None:
            self.split_ids = self._load_and_filter_original_ids_df_by_split()
        else:
            self.split_ids = split_ids

    def apply(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        """Filter the dataloader to only include ids from the desired splits
        from the original datasplit"""
        return input_df.filter(pl.col(self.id_col_name).is_in(self.split_ids))

    def _load_and_filter_original_ids_df_by_split(self) -> pl.Series:
        split_names: list[SplitName] = []

        for split_name in self.splits_to_keep:  # type: ignore already guarded in init
            match split_name:
                case "train":
                    split_names.append(SplitName.TRAIN)
                case "val":
                    split_names.append(SplitName.VALIDATION)
                case "test":
                    split_names.append(SplitName.TEST)
                case _:  # pyright: ignore [reportUnnecessaryComparison]
                    raise ValueError(
                        f"Splitname {split_name} is not allowed, try from ['train', 'test', 'val']",
                    )
        return (
            pl.concat(
                [
                    load_stratified_by_outcome_split_ids(split).frame
                    for split in split_names
                ],
            )
            .collect()
            .get_column("dw_ek_borger")
        )
