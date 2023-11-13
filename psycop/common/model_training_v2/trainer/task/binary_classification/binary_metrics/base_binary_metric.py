from __future__ import annotations

from typing import TYPE_CHECKING, runtime_checkable

from psycop.common.model_training_v2.trainer.task.base_metric import BaseMetric

if TYPE_CHECKING:
    import pandas as pd

    from psycop.common.model_training_v2.trainer.task.base_metric import (
        CalculatedMetric,
    )
    from psycop.common.model_training_v2.trainer.task.binary_classification.binary_classification_pipeline import (
        PredProbaSeries,
    )


@runtime_checkable
class BinaryMetric(BaseMetric):
    def calculate(
        self,
        y_true: pd.Series[int],
        y_pred: PredProbaSeries,
    ) -> CalculatedMetric:
        ...
