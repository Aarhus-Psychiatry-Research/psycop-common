from typing import Iterable, Literal

import polars as pl
from psycop.common.data_inspection.visits_by_hospital_units.make_geographical_split import (
    GEOGRAPHICAL_SPLIT_PATH,
)
from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader
from psycop.common.model_training_v2.trainer.data.data_filters.base_data_filter import (
    BaselineDataFilter,
)


def subset_by_timestamp(
    prediction_times_df: pl.DataFrame,
    cuttoff_df: pl.DataFrame,
    id_col_name: str,
    timestamp_col_name: str,
) -> pl.DataFrame:
    x = prediction_times_df.join(cuttoff_df, on=id_col_name, how="left")
    return x.filter(
        pl.col(timestamp_col_name)
        < pl.col("cutoff_timestamp") | pl.col("cutoff_timestamp").is_null()
    )


class GeographyDataFilter(BaselineDataFilter):
    def __init__(
        self,
        regions: Iterable[Literal["vest", "midt", "øst"]],
        id_col_name: str,
        timestamp_col_name: str,
    ):
        self.splits = regions
        self.id_col_name = id_col_name
        self.timestamp_col_name = timestamp_col_name

        self.filtered_geography_id_df = self._prepare_geography_df()

    def apply(self, dataloader: BaselineDataLoader) -> pl.LazyFrame:
        filtered_df = dataloader.load().join(
            self.filtered_geography_id_df, on=self.id_col_name, how="inner"
        )
        # TODO: only keep visits at first region

    def _prepare_geography_df(self) -> pl.LazyFrame:
        """Keep only the ids from the desired regions and rename dw_ek_borger
        to match the id_col_name of the incoming dataloader"""
        return (
            pl.scan_parquet(GEOGRAPHICAL_SPLIT_PATH)
            .filter(pl.col("region").is_in(self.splits))
            .rename({"dw_ek_borger": self.id_col_name})
        )
