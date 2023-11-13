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
    from psycop.common.model_training_v2.trainer.task.binary_classification.binary_eval_dataset import (
        BinaryEvalDataset,
    )

    pass


@BaselineRegistry.metrics.register("binary_auroc")
class BinaryAUROC(BinaryMetric):
    def __init__(self) -> None:
        pass

    def calculate(
        self,
        eval_dataset: BinaryEvalDataset,
    ) -> CalculatedMetric:
        return CalculatedMetric(
            name="BinaryAUROC",
            value=float(
                roc_auc_score(
                    y_true=eval_dataset.df.select(eval_dataset.y_col),
                    y_score=eval_dataset.df.select(eval_dataset.y_hat_col),
                ),
            ),
        )
