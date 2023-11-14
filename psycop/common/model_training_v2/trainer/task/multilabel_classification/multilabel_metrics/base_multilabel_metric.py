from typing import Protocol, runtime_checkable

from psycop.common.model_training_v2.trainer.task.base_metric import (
    CalculatedMetric,
)
from psycop.common.model_training_v2.trainer.task.eval_dataset_base import (
    BaseEvalDataset,
)


@runtime_checkable
class MultilabelMetric(Protocol):
    def calculate(
        self,
        eval_dataset: BaseEvalDataset,
    ) -> CalculatedMetric:
        ...
