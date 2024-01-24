from abc import ABC
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import pandas as pd
import polars as pl

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.loggers.base_logger import BaselineLogger

from ...loggers.supports_logger import SupportsLoggerMixin
from .step import PresplitStep


class PreprocessingPipeline(ABC, SupportsLoggerMixin):
    steps: Sequence[PresplitStep]

    def apply(self, data: pl.LazyFrame) -> pd.DataFrame:
        ...


@BaselineRegistry.preprocessing.register("baseline_preprocessing_pipeline")
class BaselinePreprocessingPipeline(PreprocessingPipeline):
    def __init__(self, *args: PresplitStep) -> None:
        self.steps = list(args)

    def _get_column_stats_string(self, data: pl.LazyFrame) -> str:
        return f"""
n_cols: {len(data.columns)}
Columns: {data.columns}"""

    def apply(self, data: pl.LazyFrame) -> pd.DataFrame:
        self.logger.info(self._get_column_stats_string(data))

        for step in self.steps:
            data = step.apply(data)

        self.logger.info(self._get_column_stats_string(data))
        return data.collect().to_pandas()
