from collections.abc import Sequence
from typing import Protocol

import polars as pl

from psycop.common.model_training_v2.classifier_pipelines.binary_classification_pipeline import (
    BinaryClassificationPipeline,
    PredProbaSeries,
)
from psycop.common.model_training_v2.problem_type.eval_dataset_base import (
    BinaryEvalDataset,
)
from psycop.common.model_training_v2.training_method.base_training_method import (
    TrainingResult,
)

from ..classifier_pipelines.multiclass_classification_pipeline import (
    MulticlassClassificationPipeline,
)
from ..metrics.binary_metrics.base_binary_metric import BinaryMetric
from ..metrics.multilabel_metrics.base import MultilabelMetric
from ..presplit_preprocessing.polars_frame import PolarsFrame


class ProblemType(Protocol):
    def train(self, x: PolarsFrame, y: PolarsFrame) -> None:
        """Train the model"""
        ...

    def evaluate(self, x: PolarsFrame, y: PolarsFrame) -> TrainingResult:
        ...

    def predict_proba(self, x: PolarsFrame) -> pl.Series:
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
        y: pl.Series,
    ) -> None:
        self.pipe.fit(X, y)

    def predict_proba(self, x: PolarsFrame) -> PredProbaSeries:
        return self.pipe.predict_proba(x)

    def evaluate(self, x: PolarsFrame, y: pl.Series) -> TrainingResult:
        if isinstance(x, pl.LazyFrame):
            x = x.collect()
        y_hat_probs = self.pipe.predict_proba(x)

        df = x.with_columns(y_hat_probs, y)
        eval_dataset = BinaryEvalDataset(
            pred_time_uuids="pred_time_uuids",  # need to get this column from somewhere!
            y_hat_probs=str(y_hat_probs.name),
            y=y.name,
            df=df,
        )
        main_metric = eval_dataset.calculate_metrics([self.main_metric])[0]
        supplementary_metrics = (
            eval_dataset.calculate_metrics(self.supplementary_metrics)
            if self.supplementary_metrics is not None
            else None
        )

        return TrainingResult(
            main_metric=main_metric,
            supplementary_metrics=supplementary_metrics,
            eval_dataset=eval_dataset,
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
        X: PolarsFrame,
        y: PolarsFrame,
    ) -> float:
        ...
