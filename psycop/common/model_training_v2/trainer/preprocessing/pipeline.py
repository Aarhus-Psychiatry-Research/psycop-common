from abc import ABC, abstractmethod
from collections.abc import Sequence

import pandas as pd
import polars as pl
import rich.pretty as pretty

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry

from ...loggers.supports_logger import SupportsLoggerMixin
from .step import PresplitStep


class PreprocessingPipeline(ABC, SupportsLoggerMixin):
    steps: Sequence[PresplitStep]

    @abstractmethod
    def apply(self, data: pl.LazyFrame) -> pd.DataFrame: ...


@BaselineRegistry.preprocessing.register("baseline_preprocessing_pipeline")
class BaselinePreprocessingPipeline(PreprocessingPipeline):
    def __init__(self, *args: PresplitStep, eager: bool = False) -> None:
        self.steps = list(args)
        self.eager = eager

    def _get_column_stats_string(self, data: pl.LazyFrame) -> str:
        return f"""
    Columns: {pretty.pretty_repr(sorted(data.columns), max_width=100)}
    n_cols: {len(data.columns)}"""

    def apply(self, data: pl.LazyFrame) -> pd.DataFrame:
        self.logger.info(
            f"Column stats before preprocessing: {self._get_column_stats_string(data)}"
        )

        for step in self.steps:
            data = step.apply(data)
            if self.eager:
                collected_data = data.collect()
                self.logger.info(
                    f"Number of rows after {step.__class__.__name__}: {len(collected_data)}"
                )

        self.logger.info(f"Column stats after preprocessing: {self._get_column_stats_string(data)}")

        preprocessed_data = data.collect().to_pandas()
        self.logger.info(f"Number of rows after preprocessing: {len(preprocessed_data)}")

        return preprocessed_data
