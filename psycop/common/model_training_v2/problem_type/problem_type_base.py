from collections.abc import Sequence
from typing import Protocol

from psycop.common.model_training_v2.classifier_pipelines.binary_classification_pipeline import (
    BinaryClassificationPipeline,
)

from ..classifier_pipelines.multiclass_classification_pipeline import (
    MulticlassClassificationPipeline,
)
from ..metrics.binary_metrics.base_binary_metric import BinaryMetric
from ..metrics.multilabel_metrics.base_multilabel_metric import MultilabelMetric
from ..presplit_preprocessing.polars_frame import PolarsFrame


class ProblemType(Protocol):
    def train(self, x: PolarsFrame, y: PolarsFrame) -> float:
        ...


class BinaryClassification:
    def __init__(
        self,
        pipe: BinaryClassificationPipeline,
        main_metric: BinaryMetric,
        supplementary_metrics: Sequence[BinaryMetric] | None = None,
    ):
        self.pipe = pipe
        self.main_metric = main_metric
        self.supplementary_metrics = supplementary_metrics

    def train(
        self,
        X: PolarsFrame,
        y: PolarsFrame,
    ) -> float:
        ...


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
        X: PolarsFrame,
        y: PolarsFrame,
    ) -> float:
        ...
