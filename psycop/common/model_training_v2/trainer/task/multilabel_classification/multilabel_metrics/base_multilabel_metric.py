import pandas as pd

from psycop.common.model_training_v2.trainer.task.base_metric import (
    BaseMetric,
    CalculatedMetric,
)
from psycop.common.model_training_v2.trainer.task.binary_classification.binary_classification_pipeline import (
    PredProbaSeries,
)


class MultilabelMetric(BaseMetric):
    def calculate(
        self,
        y_true: pd.Series[int],
        y_pred: PredProbaSeries,
    ) -> CalculatedMetric:
        ...
