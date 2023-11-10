from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd

    from psycop.common.model_training_v2.trainer.task.base_metric import (
        CalculatedMetric,
    )
    from psycop.common.model_training_v2.trainer.task.binary_classification.binary_classification_pipeline import (
        PredProbaSeries,
    )


@runtime_checkable
class BinaryMetric(Protocol):
    def calculate(
        self,
        y_true: pd.Series[int],
        y_pred: PredProbaSeries,
    ) -> CalculatedMetric:
        ...
