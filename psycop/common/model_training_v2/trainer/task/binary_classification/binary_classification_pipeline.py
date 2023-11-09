import pandas as pd
import polars as pl
from sklearn.pipeline import Pipeline

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.preprocessing.polars_frame import (
    PolarsFrame,
)

PredProbaSeries = pd.Series  # name should be "y_hat_probs", series of floats


@BaselineRegistry.baseline_task_pipelines.register("binary_classification_pipeline")
class BinaryClassificationPipeline:
    def __init__(self, sklearn_pipe: Pipeline):
        self.pipe = sklearn_pipe

    def fit(self, x: PolarsFrame, y: pl.Series) -> None:
        if isinstance(x, pl.LazyFrame):
            x = x.collect()
        self.pipe.fit(X=x.to_pandas(), y=y)

    def predict_proba(self, x: PolarsFrame) -> PredProbaSeries:
        """Returns the predicted probabilities of the `1`
        class"""
        if isinstance(x, pl.LazyFrame):
            x = x.collect()
        pred_probs = self.pipe.predict_proba(x.to_pandas())[:, 1]
        return pd.Series(pred_probs, name="y_hat_probs")
