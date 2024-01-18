from collections.abc import Sequence
from typing import Protocol

from psycop.common.model_training_v2.trainer.task.model_step import (
    ModelStep,
)
from psycop.common.types.polarsframe import PolarsFrame

from ..base_pipeline import BasePipeline


class MultilabelClassificationPipeline(BasePipeline, Protocol):
    def __init__(self, steps: Sequence[ModelStep]) -> None:
        ...

    def fit(self, X: PolarsFrame, y: PolarsFrame) -> None:
        ...

    def predict_proba(self, X: PolarsFrame) -> PolarsFrame:
        ...
