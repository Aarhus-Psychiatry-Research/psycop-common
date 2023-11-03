from collections.abc import Sequence
from typing import Protocol

from ..loggers.base_logger import BaselineLogger
from .polars_frame import PolarsFrame
from .step import PresplitStep


class PreprocessingPipeline(Protocol):
    def __init__(self, steps: Sequence[PresplitStep], logger: BaselineLogger):
        ...

    def apply(self, data: PolarsFrame) -> PolarsFrame:
        ...


class BaselinePreprocessingPipeline:
    def __init__(self, steps: Sequence[PresplitStep]) -> None:
        self.steps = steps

    def apply(self, data: PolarsFrame) -> PolarsFrame:
        for step in self.steps:
            data = step.apply(data)
        return data