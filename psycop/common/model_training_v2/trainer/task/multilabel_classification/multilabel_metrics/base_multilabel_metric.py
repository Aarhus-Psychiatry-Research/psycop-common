import pandas as pd

from psycop.common.model_training_v2.trainer.task.base_metric import (
    BaselineMetric,
    CalculatedMetric,
    PredProbaSeries,
)


class MultilabelMetric(BaselineMetric):
    def calculate(
        self,
        y: pd.Series,  # type: ignore
        y_hat_prob: PredProbaSeries,
        name_prefix: str | None = None,
    ) -> CalculatedMetric: ...
