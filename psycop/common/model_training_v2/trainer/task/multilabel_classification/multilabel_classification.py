from collections.abc import Sequence

import pandas as pd

from psycop.common.model_training_v2.trainer.base_trainer import (
    TrainingResult,
)
from psycop.common.model_training_v2.trainer.preprocessing.polars_frame import (
    PolarsFrame,
)
from psycop.common.model_training_v2.trainer.task.base_task import (
    BaselineTask,
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
        x: PolarsFrame,
        y: PolarsFrame,
    ):
        ...

    def predict_proba(self, x: PolarsFrame) -> pd.Series[float]:
        ...

    def evaluate(self, x: PolarsFrame, y: PolarsFrame) -> TrainingResult:
        ...
