from psycop.common.model_training_v2.trainer.task.base_metric import (
    BaseMetric,
    CalculatedMetric,
)
from psycop.common.model_training_v2.trainer.task.eval_dataset_base import (
    BaseEvalDataset,
)


class MultilabelMetric(BaseMetric):
    def calculate(
        self,
        eval_dataset: BaseEvalDataset,
    ) -> CalculatedMetric:
        ...
