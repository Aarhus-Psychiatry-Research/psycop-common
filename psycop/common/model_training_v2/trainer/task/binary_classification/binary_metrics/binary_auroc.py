from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.metrics import roc_auc_score

from psycop.common.model_training_v2.training_method.problem_type.base_metric import (
    CalculatedMetric,
)
from psycop.common.model_training_v2.training_method.problem_type.binary_classification.binary_metrics.base_binary_metric import (
    BinaryMetric,
)

if TYPE_CHECKING:
    import pandas as pd

    from psycop.common.model_training_v2.training_method.problem_type.binary_classification.binary_classification_pipeline import (
        PredProbaSeries,
    )


class BinaryAUROC(BinaryMetric):
    def calculate(
        self,
        y_true: pd.Series[int],
        y_pred: PredProbaSeries,
    ) -> CalculatedMetric:
        return CalculatedMetric(
            name="BinaryAUROC",
            value=float(roc_auc_score(y_true=y_true, y_score=y_pred)),
        )
