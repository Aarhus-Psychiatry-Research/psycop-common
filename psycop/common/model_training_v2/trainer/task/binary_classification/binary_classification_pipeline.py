from collections.abc import Sequence

import pandas as pd
import polars as pl
from sklearn.pipeline import Pipeline

from psycop.common.model_training_v2.training_method.preprocessing.polars_frame import (
    PolarsFrame,
)
from psycop.common.model_training_v2.training_method.problem_type.model_step import (
    ModelStep,
)

PredProbaSeries = pd.Series  # name should be "y_hat_probs", series of floats


class BinaryClassificationPipeline:
    def __init__(self, steps: Sequence[ModelStep]):
        self.pipe = Pipeline(steps=steps)

    def fit(self, x: PolarsFrame, y: pl.Series) -> None:
        if isinstance(x, pl.LazyFrame):
            x = x.collect()
        self.pipe.fit(x.to_pandas(), y)

    def predict_proba(self, x: PolarsFrame) -> PredProbaSeries:
        """Returns the predicted probabilities of the `1`
        class"""
        if isinstance(x, pl.LazyFrame):
            x = x.collect()
        pred_probs = self.pipe.predict_proba(x.to_pandas())[:, 1]
        return pd.Series(pred_probs, name="y_hat_probs")
