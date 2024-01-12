from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import polars as pl

from psycop.common.feature_generation.loaders.raw.load_ids import (
    SplitName,
    load_stratified_by_outcome_split_ids,
)
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.preprocessing.step import PresplitStep


@BaselineRegistry.data_filters.register("id_data_filter")
@dataclass(frozen=True)
class FilterByEntityID(PresplitStep):
    """Filter the data to only include ids in split_ids"""

    split_ids: Sequence[int] | pl.Series
    id_col_name: str = "dw_ek_borger"

    def apply(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        return input_df.filter(pl.col(self.id_col_name).is_in(self.split_ids))


@BaselineRegistry.data_filters.register("outcomestratified_split_filter")
@dataclass(frozen=True)
class FilterByOutcomeStratifiedSplits(PresplitStep):
    """Filter the data to only include ids from the splits stratified by outcome, for the given split_to_keep."""

    splits_to_keep: Sequence[Literal["train", "val", "test"]]
    id_col_name: str = "dw_ek_borger"

    def apply(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        """Filter the dataloader to only include ids from the desired splits
        from the original datasplit"""
        split_ids = self._load_and_filter_original_ids_df_by_split()
        return input_df.filter(pl.col(self.id_col_name).is_in(split_ids))

    def _load_and_filter_original_ids_df_by_split(self) -> pl.Series:
        split_names: list[SplitName] = []

        for split_name in self.splits_to_keep:
            match split_name:
                case "train":
                    split_names.append(SplitName.TRAIN)
                case "val":
                    split_names.append(SplitName.VALIDATION)
                case "test":
                    split_names.append(SplitName.TEST)

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
