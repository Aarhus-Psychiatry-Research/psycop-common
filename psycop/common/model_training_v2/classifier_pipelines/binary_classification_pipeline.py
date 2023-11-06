from collections.abc import Sequence

import polars as pl
from sklearn.pipeline import Pipeline

from psycop.common.model_training_v2.classifier_pipelines.model_step import ModelStep
from psycop.common.model_training_v2.presplit_preprocessing.polars_frame import (
    PolarsFrame,
)


class BinaryClassificationPipeline:
    def __init__(self, steps: Sequence[ModelStep]):
        self.pipe = Pipeline(steps=steps)

    def fit(self, X: PolarsFrame, y: pl.Series) -> None:
        if isinstance(X, pl.LazyFrame):
            X = X.collect()
        self.pipe.fit(X.to_pandas(), y)

    def predict_proba(self, X: PolarsFrame) -> pl.Series:
        """Returns the predicted probabilities of the `1`
        class"""
        if isinstance(X, pl.LazyFrame):
            X = X.collect()
        pred_probs = self.pipe.predict_proba(X.to_pandas())[:, 1]
        return pl.Series("y_hat_probs", pred_probs)
