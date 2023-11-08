from typing import Protocol

from psycop.common.model_training_v2.trainer.preprocessing.polars_frame import (
    PolarsFrame,
)
from psycop.common.model_training_v2.trainer.task.base_metric import (
    CalculatedMetric,
)


class MultilabelMetric(Protocol):
    def calculate(self, y_true: PolarsFrame, y_pred: PolarsFrame) -> CalculatedMetric:
        ...
