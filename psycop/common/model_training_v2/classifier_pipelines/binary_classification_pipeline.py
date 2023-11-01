from collections.abc import Sequence

import polars as pl

from psycop.common.model_training_v2.classifier_pipelines.model_step import ModelStep
from psycop.common.model_training_v2.presplit_preprocessing.polars_frame import (
    PolarsFrame,
)


class BinaryClassificationPipeline:
    def __init__(self, steps: Sequence[ModelStep]):
        ...

    def fit(self, X: PolarsFrame, y: PolarsFrame) -> None:
        ...

    def predict_proba(self, X: PolarsFrame) -> pl.Series:
        ...
