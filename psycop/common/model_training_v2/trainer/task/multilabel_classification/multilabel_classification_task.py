from collections.abc import Sequence

import pandas as pd

from psycop.common.model_training_v2.trainer.task.base_task import (
    BaselineTask,
)
from psycop.common.model_training_v2.trainer.task.multilabel_classification.multilabel_classification_pipeline import (
    MultilabelClassificationPipeline,
)
from psycop.common.model_training_v2.trainer.task.multilabel_classification.multilabel_metrics.base_multilabel_metric import (
    MultilabelMetric,
)


class MultilabelClassificationTask(BaselineTask):
    def __init__(
        self,
        pipe: MultilabelClassificationPipeline,
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
