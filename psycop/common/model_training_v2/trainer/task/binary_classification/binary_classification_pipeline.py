import pandas as pd
from sklearn.pipeline import Pipeline

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.task.base_metric import PredProbaSeries


@BaselineRegistry.task_pipelines.register("binary_classification_pipeline")
class BinaryClassificationPipeline:
    def __init__(self, sklearn_pipe: Pipeline):
        self.pipe = sklearn_pipe

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:  # type: ignore
        self.pipe.fit(X=x, y=y)

    def predict_proba(self, x: pd.DataFrame) -> PredProbaSeries:
        """Returns the predicted probabilities of the `1`
        class"""
        pred_probs = self.pipe.predict_proba(x)[:, 1]
        return pd.Series(pred_probs, name="y_hat_probs")
