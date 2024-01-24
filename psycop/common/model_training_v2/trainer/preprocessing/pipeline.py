from collections.abc import Sequence
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

    def apply(self, data: pl.LazyFrame) -> pd.DataFrame:
        ...


@BaselineRegistry.preprocessing.register("baseline_preprocessing_pipeline")
class BaselinePreprocessingPipeline(PreprocessingPipeline):
    def __init__(self, *args: PresplitStep, logger: BaselineLogger | None) -> None:
        self.steps = list(args)
        self.logger = logger

    def _get_column_stats_string(self, data: pl.LazyFrame) -> str:
        return f"""
n_cols: {len(data.columns)}
Columns: {data.columns}"""

    def apply(self, data: pl.LazyFrame) -> pd.DataFrame:
        if self.logger:
            self.logger.info(self._get_column_stats_string(data))

        for step in self.steps:
            data = step.apply(data)

        if self.logger:
            self.logger.info(self._get_column_stats_string(data))
        return data.collect().to_pandas()
