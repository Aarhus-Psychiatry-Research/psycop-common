from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader


import polars as pl

@BaselineRegistry.data.register("test_data_filter")
class TestDataFilter:
    def apply(self, dataloader: BaselineDataLoader):
        return dataloader.load().filter(pl.col("dw_ek_borger") == 1)
    