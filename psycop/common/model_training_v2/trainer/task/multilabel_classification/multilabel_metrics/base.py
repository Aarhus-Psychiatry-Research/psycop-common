from typing import Protocol

from psycop.common.model_training_v2.training_method.preprocessing.polars_frame import PolarsFrame


class MultilabelMetric(Protocol):
    def __call__(self, y_true: PolarsFrame, y_pred: PolarsFrame) -> float:
        ...
