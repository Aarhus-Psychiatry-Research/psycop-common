from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import polars as pl

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.preprocessing.step import (
    PresplitStep,
)
from psycop.common.model_training_v2.trainer.preprocessing.steps.geographical_split.make_geographical_split import (
    get_regional_split_df,
)

from .....feature_generation.loaders.raw.load_ids import (
    SplitName,
    load_stratified_by_outcome_split_ids,
)


@BaselineRegistry.preprocessing.register("regional_data_filter")
@dataclass(frozen=True)
class RegionalFilter(PresplitStep):
    """Filter data to only include ids from the desired regions. Removes
    predictions times after the patient has moved to a different region to
    avoid target leakage. If regional_move_df is None, the standard regional
    move dataframe is loaded .

    Args:
        regions_to_keep: The regions to keep
        id_col_name (str, optional): The name of the id column. Defaults to "dw_ek_borger".
        timestamp_col_name (str, optional): The name of the timestamp column. Defaults to "timestamp".
        regional_move_df (pl.LazyFrame | None, optional): The dataframe containing the regional move data. Defaults to None.
            If supplied, should contain "dw_ek_borger", a region col and a timestamp column indicating when the patient moved.
        region_col_name (str, optional): The name of the region column in regional_move_df. Defaults to "region".
        timestamp_cutoff_col_name (str, optional): The name of the timestamp column in regional_move_df to
            be used for dropping rows after this time. Defaults to "first_regional_move_timestamp".
    """

    splits_to_keep: Sequence[Literal["train", "test", "val"]]
    id_col_name: str = "dw_ek_borger"
    timestamp_col_name: str = "timestamp"
    region_col_name: str = "region"

    regional_move_df: pl.LazyFrame | None = None
    timestamp_cutoff_col_name: str = "first_regional_move_timestamp"

    def apply(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        """Only include data from the first region a patient visits, to avoid
        target leakage."""

        regional_move_df = (
            get_regional_split_df().select(
                "dw_ek_borger",
                "region",
                "first_regional_move_timestamp",
            )
            if self.regional_move_df is None
            else self.regional_move_df
        )

        filtered_regional_move_df = self._filter_regional_move_df_by_regions(
            regional_move_df,
        )

        return (
            input_df.join(filtered_regional_move_df, on=self.id_col_name, how="inner")
            .filter(
                pl.col(self.timestamp_col_name)
                < pl.col(self.timestamp_cutoff_col_name),
            )
            .drop(columns=[self.region_col_name, self.timestamp_cutoff_col_name])
        )

    def _filter_regional_move_df_by_regions(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Keep only the ids from the desired regions and rename dw_ek_borger
        to match the id_col_name of the incoming dataloader"""
        splits2region = {"train": "øst", "val": "vest", "test": "midt"}
        regions_to_keep = {splits2region[split] for split in self.splits_to_keep}

        return df.rename({"dw_ek_borger": self.id_col_name}).filter(
            pl.col(self.region_col_name).is_in(regions_to_keep),
        )


@BaselineRegistry.preprocessing.register("id_data_filter")
@dataclass(frozen=True)
class FilterByEntityID(PresplitStep):
    """Filter the data to only include ids in split_ids"""

    split_ids: Sequence[int] | pl.Series
    id_col_name: str = "dw_ek_borger"

    def apply(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        return input_df.filter(pl.col(self.id_col_name).is_in(self.split_ids))


@BaselineRegistry.preprocessing.register("outcomestratified_split_filter")
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