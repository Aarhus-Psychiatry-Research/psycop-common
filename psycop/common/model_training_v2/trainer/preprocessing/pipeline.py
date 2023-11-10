from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.loggers.base_logger import BaselineLogger

from .polars_frame import PolarsFrame
from .step import PresplitStep


@runtime_checkable
class PreprocessingPipeline(Protocol):
    def __init__(self, steps: Sequence[PresplitStep], logger: BaselineLogger):
        ...

    def apply(self, data: PolarsFrame) -> PolarsFrame:
        ...


@BaselineRegistry.preprocessing.register("baseline_preprocessing_pipeline")
class BaselinePreprocessingPipeline(
    PreprocessingPipeline,
):  # TODO: #406 Does registering this into a registry remove protocol checking? E.g. the __init__ method does not adhered to PreprocessingPipeline
    def __init__(self, *args: PresplitStep) -> None:
        self.steps = list(args)

    def apply(self, data: PolarsFrame) -> PolarsFrame:
        for step in self.steps:
            data = step.apply(data)
        return data
