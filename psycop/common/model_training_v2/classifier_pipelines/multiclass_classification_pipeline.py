from collections.abc import Sequence
from typing import Protocol

from psycop.common.model_training_v2.classifier_pipelines.model_step import ModelStep

from ..presplit_preprocessing.polars_frame import PolarsFrame


class MulticlassClassificationPipeline(Protocol):
    def __init__(self, steps: Sequence[ModelStep]) -> None:
        ...

    def fit(self, X: PolarsFrame, y: PolarsFrame) -> None:
        ...

    def predict_proba(self, X: PolarsFrame) -> PolarsFrame:
        ...
