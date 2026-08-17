from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.metrics import roc_auc_score, average_precision_score

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.task.base_metric import CalculatedMetric
from psycop.common.model_training_v2.trainer.task.binary_classification.binary_metrics.base_binary_metric import (
    BinaryMetric,
)

if TYPE_CHECKING:
    import pandas as pd

    from psycop.common.model_training_v2.trainer.task.base_metric import PredProbaSeries

@BaselineRegistry.metrics.register("binary_average_precision")
class BinaryAveragePrecision(BinaryMetric):
    """
    Area under the precision-recall curve, computed as the weighted mean of
    precisions at each threshold (i.e. Average Precision / AP score).
    Equivalent to the AUC of the PR curve.
    """

    def __init__(self) -> None:
        pass

    def calculate(
        self,
        y: pd.Series,  # type: ignore
        y_hat_prob: PredProbaSeries,
        name_prefix: str | None = None,
    ) -> CalculatedMetric:
        return CalculatedMetric(
            name=f"{name_prefix}_BinaryAveragePrecision" if name_prefix else "BinaryAveragePrecision",
            value=float(average_precision_score(y_true=y, y_score=y_hat_prob)),
        )
