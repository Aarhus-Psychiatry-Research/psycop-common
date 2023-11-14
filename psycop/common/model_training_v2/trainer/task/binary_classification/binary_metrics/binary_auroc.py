from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.metrics import roc_auc_score

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.task.base_metric import (
    CalculatedMetric,
)
from psycop.common.model_training_v2.trainer.task.binary_classification.binary_metrics.base_binary_metric import (
    BinaryMetric,
)

if TYPE_CHECKING:
    import pandas as pd

    from psycop.common.model_training_v2.trainer.task.base_metric import (
        PredProbaSeries,
    )


@BaselineRegistry.metrics.register("binary_auroc")
class BinaryAUROC(BinaryMetric):
    def __init__(self) -> None:
        pass

    def calculate(
        self,
        y: pd.Series,  # type: ignore
        y_hat_prob: PredProbaSeries,
    ) -> CalculatedMetric:
        return CalculatedMetric(
            name="BinaryAUROC",
            value=float(
                roc_auc_score(
                    y_true=y,
                    y_score=y_hat_prob,
                ),
            ),
        )
