from dataclasses import dataclass

import pandas as pd
from sklearn.pipeline import Pipeline

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.task.base_metric import PredProbaSeries

from ..base_pipeline import BasePipeline


@BaselineRegistry.task_pipelines.register("binary_classification_pipeline")
@dataclass
class BinaryClassificationPipeline(BasePipeline):
    sklearn_pipe: Pipeline

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:  # type: ignore
        self.sklearn_pipe.fit(X=x, y=y)

    def predict_proba(self, x: pd.DataFrame) -> PredProbaSeries:
        """Returns the predicted probabilities of the `1`
        class"""
        pred_probs = self.sklearn_pipe.predict_proba(x)[:, 1]
        return pd.Series(pred_probs, name="y_hat_prob")
