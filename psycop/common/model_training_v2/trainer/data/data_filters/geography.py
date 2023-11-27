

from typing import Iterable, Literal
import polars as pl
from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader
from psycop.common.model_training_v2.trainer.data.data_filters.base_data_filter import BaselineDataFilter


class GeographyDataFilter(BaselineDataFilter):
    def __init__(self, split: Iterable[Literal["vest", "midt", "øst"]], id_col_name: str):
        self.split = split
        self.id_col_name = id_col_name

        # TODO: set correct ids
        self.geography_ids = {
            "vest": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "midt": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20"],
            "øst": ["21", "22", "23", "24", "25", "26", "27", "28", "29", "30"],
        }

    def apply(self, dataloader: BaselineDataLoader) -> pl.LazyFrame:
        ids_to_keep = [ids for split in self.split for ids in self.geography_ids[split]]
        # TODO: only keep visits at first region
        return dataloader.load().filter(pl.col("id_col_name").is_in(ids_to_keep))