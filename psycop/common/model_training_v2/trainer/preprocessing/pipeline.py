from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import pandas as pd
import polars as pl

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.loggers.base_logger import BaselineLogger

from .polars_frame import PolarsFrame
from .step import PresplitStep


@runtime_checkable
class PreprocessingPipeline(Protocol):
    def __init__(self, steps: Sequence[PresplitStep], logger: BaselineLogger):
        ...

    def apply(self, data: PolarsFrame) -> pd.DataFrame:
        ...


@BaselineRegistry.preprocessing.register("baseline_preprocessing_pipeline")
class BaselinePreprocessingPipeline(
    PreprocessingPipeline,
):
    def __init__(self, *args: PresplitStep) -> None:
        self.steps = list(args)

    def apply(self, data: PolarsFrame) -> pd.DataFrame:
        for step in self.steps:
            data = step.apply(data)

        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        return data.to_pandas()
