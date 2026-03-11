from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.metrics import roc_auc_score

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.task.base_metric import CalculatedMetric
from psycop.common.model_training_v2.trainer.task.binary_classification.binary_metrics.base_binary_metric import (
    BinaryMetric,
)

if TYPE_CHECKING:
    import pandas as pd

    from psycop.common.model_training_v2.trainer.task.base_metric import PredProbaSeries


@BaselineRegistry.metrics.register("binary_auroc")
class BinaryAUROC(BinaryMetric):
    def __init__(self) -> None:
        pass

    def calculate(
        self,
        y: pd.Series,  # type: ignore
        y_hat_prob: PredProbaSeries,
        name_prefix: str | None = None,
    ) -> CalculatedMetric:
        return CalculatedMetric(
            name=f"{name_prefix}_BinaryAUROC" if name_prefix else "BinaryAUROC",
            value=float(roc_auc_score(y_true=y, y_score=y_hat_prob)),
        )


@BaselineRegistry.metrics.register("concentrated_binary_auroc")
class ConcentratedBinaryAUROC(BinaryMetric):
    """
    max_fpr allows for calculation of concentrated AUROC. If max_fpr is None, calculates standard AUROC.
    """

    def __init__(self, max_fpr: None | float = 0.2) -> None:
        self.max_fpr = max_fpr

    def calculate(
        self,
        y: pd.Series,  # type: ignore
        y_hat_prob: PredProbaSeries,
        name_prefix: str | None = None,
    ) -> CalculatedMetric:
        return CalculatedMetric(
            name=f"{name_prefix}_ConcentratedBinaryAUROC"
            if name_prefix
            else "ConcentratedBinaryAUROC",
            value=float(roc_auc_score(y_true=y, y_score=y_hat_prob, max_fpr=self.max_fpr)),
        )
