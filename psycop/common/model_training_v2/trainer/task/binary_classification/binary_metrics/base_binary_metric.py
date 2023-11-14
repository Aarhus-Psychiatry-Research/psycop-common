from __future__ import annotations

from typing import TYPE_CHECKING

from psycop.common.model_training_v2.trainer.task.base_metric import BaseMetric

if TYPE_CHECKING:
    import pandas as pd

    from psycop.common.model_training_v2.trainer.task.base_metric import (
        CalculatedMetric,
    )
    from psycop.common.model_training_v2.trainer.task.binary_classification.binary_classification_pipeline import (
        PredProbaSeries,
    )


class BinaryMetric(BaseMetric):
    def calculate(
        self,
        y: pd.Series,  # type: ignore
        y_hat_prob: PredProbaSeries,
    ) -> CalculatedMetric:
        ...
