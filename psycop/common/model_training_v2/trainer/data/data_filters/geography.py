from typing import Collection, Literal

import polars as pl

from psycop.common.data_inspection.visits_by_hospital_units.make_geographical_split import (
    GEOGRAPHICAL_SPLIT_PATH,
)
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader
from psycop.common.model_training_v2.trainer.data.data_filters.base_data_filter import (
    BaselineDataFilter,
)


@BaselineRegistry.data.register("geography_data_filter")
class GeographyDataFilter(BaselineDataFilter):
    def __init__(
        self,
        regions: Collection[Literal["vest", "midt", "øst"]],
        id_col_name: str,
        timestamp_col_name: str,
    ):
        self.splits = regions
        self.id_col_name = id_col_name
        self.timestamp_col_name = timestamp_col_name

    def apply(self, dataloader: BaselineDataLoader) -> pl.LazyFrame:
        """Filter the dataloader to only include ids from the desired regions
        and remove prediction times after a move to a different region"""
        filtered_geography_id_df = self._load_and_prepare_geography_df()

        return (
            dataloader.load()
            .join(filtered_geography_id_df, on=self.id_col_name, how="inner")
            .filter(pl.col(self.timestamp_col_name) < pl.col("cutoff_timestamp"))
        )

    def _load_and_prepare_geography_df(self) -> pl.LazyFrame:
        """Keep only the ids from the desired regions and rename dw_ek_borger
        to match the id_col_name of the incoming dataloader"""
        return (
            pl.scan_parquet(GEOGRAPHICAL_SPLIT_PATH)
            .filter(pl.col("region").is_in(self.splits))
            .rename({"dw_ek_borger": self.id_col_name})
        )
