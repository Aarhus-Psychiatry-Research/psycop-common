from typing import Protocol, runtime_checkable

import pandas as pd

from psycop.common.model_training_v2.trainer.task.base_metric import (
    BaseMetric,
    CalculatedMetric,
    PredProbaSeries,
)


class MultilabelMetric(BaseMetric):
    def calculate(
        self,
        y: pd.Series,  # type: ignore
        y_hat_prob: PredProbaSeries,
    ) -> CalculatedMetric:
        ...
