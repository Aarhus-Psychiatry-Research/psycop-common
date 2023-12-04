from typing import Protocol, runtime_checkable

import polars as pl

from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader


@runtime_checkable
class BaselineDataFilter(Protocol):
    def apply(self, dataloader: BaselineDataLoader) -> pl.LazyFrame:
        ...
