from collections.abc import Sequence

import polars as pl

from psycop.common.model_training_v2.classifier_pipelines.model_step import ModelStep
from psycop.common.model_training_v2.presplit_preprocessing.polars_frame import (
    PolarsFrame,
)
from sklearn.pipeline import Pipeline


class BinaryClassificationPipeline:
    def __init__(self, steps: Sequence[ModelStep]):
        self.pipe = Pipeline(steps=steps)

    def fit(self, X: PolarsFrame, y: pl.Series) -> None:
        self.pipe.fit(X, y)

    def predict_proba(self, X: PolarsFrame) -> pl.Series:
        """Returns the predicted probabilities of the `1`
        class"""
        return self.pipe.predict_proba(X)[:, 1]
