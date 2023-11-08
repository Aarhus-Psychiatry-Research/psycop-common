from typing import Protocol

from psycop.common.model_training_v2.training_method.preprocessing.polars_frame import (
    PolarsFrame,
)
from psycop.common.model_training_v2.training_method.problem_type.base_metric import (
    CalculatedMetric,
)


class MultilabelMetric(Protocol):
    def calculate(self, y_true: PolarsFrame, y_pred: PolarsFrame) -> CalculatedMetric:
        ...
