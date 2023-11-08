from collections.abc import Sequence

import pandas as pd

from psycop.common.model_training_v2.classifier_pipelines.multiclass_classification_pipeline import (
    MulticlassClassificationPipeline,
)
from psycop.common.model_training_v2.metrics.multilabel_metrics.base import (
    MultilabelMetric,
)
from psycop.common.model_training_v2.presplit_preprocessing.polars_frame import (
    PolarsFrame,
)
from psycop.common.model_training_v2.problem_type.problem_type_base import ProblemType
from psycop.common.model_training_v2.training_method.base_training_method import (
    TrainingResult,
)


class MultilabelClassification(ProblemType):
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
