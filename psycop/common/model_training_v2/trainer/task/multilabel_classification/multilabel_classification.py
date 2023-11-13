from collections.abc import Sequence

import pandas as pd

from psycop.common.model_training_v2.trainer.task.base_task import (
    BaselineTask,
)
from psycop.common.model_training_v2.trainer.task.eval_dataset_base import (
    BaseEvalDataset,
)
from psycop.common.model_training_v2.trainer.task.multilabel_classification.multiclass_classification_pipeline import (
    MulticlassClassificationPipeline,
)
from psycop.common.model_training_v2.trainer.task.multilabel_classification.multilabel_metrics.base import (
    MultilabelMetric,
)


class MultilabelClassification(BaselineTask):
    def __init__(
        self,
        pipe: MulticlassClassificationPipeline,
        main_metric: MultilabelMetric,
        supplementary_metrics: Sequence[MultilabelMetric] | None = None,
    ):
        self.pipe = pipe
        self.metrics = main_metric
        self.supplementary_metrics = supplementary_metrics

    def train(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        y_col_name: str,
    ):
        ...

    def predict_proba(self, x: pd.DataFrame) -> pd.Series[float]:
        ...

    def construct_eval_dataset(
        self,
        df: pd.DataFrame,
        y_hat_col: str,
        y_col: str,
    ) -> BaseEvalDataset:
        ...
