from __future__ import annotations

from typing import TYPE_CHECKING

from psycop.common.model_training_v2.trainer.task.base_metric import BaselineMetric

if TYPE_CHECKING:
    import pandas as pd

    from psycop.common.model_training_v2.trainer.task.base_metric import (
        CalculatedMetric,
        PredProbaSeries,
    )


class BinaryMetric(BaselineMetric):
    def calculate(
        self,
        y: pd.Series,  # type: ignore
        y_hat_prob: PredProbaSeries,
        name_prefix: str | None = None,
    ) -> CalculatedMetric: ...
