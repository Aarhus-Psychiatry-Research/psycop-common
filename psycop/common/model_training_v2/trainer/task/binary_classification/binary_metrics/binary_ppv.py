from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.metrics import precision_score

from psycop.common.model_training.training_output.dataclasses import (
    get_predictions_for_positive_rate,
)
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.task.base_metric import CalculatedMetric
from psycop.common.model_training_v2.trainer.task.binary_classification.binary_metrics.base_binary_metric import (
    BinaryMetric,
)

if TYPE_CHECKING:
    import pandas as pd

    from psycop.common.model_training_v2.trainer.task.base_metric import PredProbaSeries


@BaselineRegistry.metrics.register("binary_ppv")
class BinaryPPV(BinaryMetric):
    def __init__(self) -> None:
        pass

    def calculate(
        self,
        y: pd.Series,  # type: ignore
        y_hat_prob: PredProbaSeries,
        name_prefix: str | None = None,
        positive_rate: float | None = None,
    ) -> CalculatedMetric:
        y_hat = get_predictions_for_positive_rate(
            desired_positive_rate=positive_rate,  # type: ignore
            y_hat_probs=y_hat_prob,  # type: ignore
        )[0]

        return CalculatedMetric(
            name=f"{name_prefix}_BinaryPPV" if name_prefix else "BinaryPPV",
            value=float(precision_score(y_true=y, y_pred=y_hat)),
        )
