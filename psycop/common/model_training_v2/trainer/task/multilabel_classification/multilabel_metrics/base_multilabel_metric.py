from typing import Protocol, runtime_checkable

import pandas as pd

from psycop.common.model_training_v2.trainer.task.base_metric import (
    CalculatedMetric,
)
from psycop.common.model_training_v2.trainer.task.binary_classification.binary_classification_pipeline import (
    PredProbaSeries,
)


@runtime_checkable
class MultilabelMetric(Protocol):
    def calculate(
        self,
        y: pd.Series,  # type: ignore
        y_hat_prob: PredProbaSeries,
    ) -> CalculatedMetric:
        ...
