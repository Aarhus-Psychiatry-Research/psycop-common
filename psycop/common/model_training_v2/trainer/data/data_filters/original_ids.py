from typing import Collection, Literal

import polars as pl
from psycop.common.feature_generation.loaders.raw.load_ids import SplitName, load_ids
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader


@BaselineRegistry.data_filters.register("original_id_data_filter")
class OriginalIDDataFilter:
    def __init__(
        self, splits: Collection[Literal["train", "val", "test"]], id_col_name: str
    ):
        self.splits = splits
        self.id_col_name = id_col_name

    def apply(self, dataloader: BaselineDataLoader) -> pl.LazyFrame:
        """Filter the dataloader to only include ids from the desired splits
        from the original datasplit"""
        original_ids = self._load_and_prepare_original_ids_df()

        return dataloader.load().filter(pl.col(self.id_col_name).is_in(original_ids))

    def _load_and_prepare_original_ids_df(self) -> pl.Series:
        split_names = []
        for split_name in self.splits:
            match split_name:
                case "train":
                    split_names.append(SplitName.TRAIN)  # noqa: PLW2901
                case "val":
                    split_names.append(SplitName.VALIDATION)  # noqa: PLW2901
                case "test":
                    split_names.append(SplitName.TEST)  # noqa: PLW2901
                case _:
                    raise ValueError(
                        f"Splitname {split_name} is not allowed, try from ['train', 'test', 'val']",
                    )
        return pl.concat(
            [pl.from_pandas(load_ids(split)) for split in split_names]
        ).get_column("dw_ek_borger")
