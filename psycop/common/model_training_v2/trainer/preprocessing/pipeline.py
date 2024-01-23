from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd
import polars as pl

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.loggers.base_logger import BaselineLogger

from .step import PresplitStep


@runtime_checkable
class PreprocessingPipeline(Protocol):
    steps: Sequence[PresplitStep]
    logger: BaselineLogger | None

    def __init__(self, steps: Sequence[PresplitStep], logger: BaselineLogger):
        ...

    def apply(self, data: pl.LazyFrame) -> pd.DataFrame:
        ...


@BaselineRegistry.preprocessing.register("baseline_preprocessing_pipeline")
@dataclass(frozen=True)
class BaselinePreprocessingPipeline(PreprocessingPipeline):
    steps: Sequence[PresplitStep]
    logger: BaselineLogger | None

    def apply(self, data: pl.LazyFrame) -> pd.DataFrame:
        if self.logger:
            self.logger.info(f"Applying preprocessing pipeline. Initial columns:\n {data.schema}")
        for step in self.steps:
            data = step.apply(data)

        return data.collect().to_pandas()
