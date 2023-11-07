from collections.abc import Sequence

from psycop.common.model_training_v2.classifier_pipelines.multiclass_classification_pipeline import (
    MulticlassClassificationPipeline,
)
from psycop.common.model_training_v2.metrics.multilabel_metrics.base import (
    MultilabelMetric,
)
from psycop.common.model_training_v2.presplit_preprocessing.polars_frame import (
    PolarsFrame,
)


class MultilabelClassification:
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
    ) -> float:
        ...
