from __future__ import annotations

from typing import TYPE_CHECKING

from psycop.common.model_training_v2.trainer.task.base_metric import BaseMetric

if TYPE_CHECKING:
    from psycop.common.model_training_v2.trainer.task.base_metric import (
        CalculatedMetric,
    )
    from psycop.common.model_training_v2.trainer.task.eval_dataset_base import (
        BaseEvalDataset,
    )


class BinaryMetric(BaseMetric):
    def calculate(
        self,
        eval_dataset: BaseEvalDataset,
    ) -> CalculatedMetric:
        ...
