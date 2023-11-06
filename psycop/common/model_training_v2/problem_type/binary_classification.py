from collections.abc import Sequence

import polars as pl

from psycop.common.model_training_v2.classifier_pipelines.binary_classification_pipeline import (
    BinaryClassificationPipeline,
    PredProbaSeries,
)
from psycop.common.model_training_v2.metrics.binary_metrics.base_binary_metric import (
    BinaryMetric,
)
from psycop.common.model_training_v2.presplit_preprocessing.polars_frame import (
    PolarsFrame,
)
from psycop.common.model_training_v2.problem_type.eval_dataset_base import (
    BinaryEvalDataset,
)
from psycop.common.model_training_v2.training_method.base_training_method import (
    TrainingResult,
)


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
        x: PolarsFrame,
        y: pl.Series,
    ) -> None:
        self.pipe.fit(x=x, y=y)

    def predict_proba(self, x: PolarsFrame) -> PredProbaSeries:
        return self.pipe.predict_proba(x)

    def evaluate(self, x: PolarsFrame, y: pl.Series) -> TrainingResult:
        if isinstance(x, pl.LazyFrame):
            x = x.collect()
        y_hat_probs = self.pipe.predict_proba(x)

        df = x.with_columns(pl.Series(y_hat_probs).alias(str(y_hat_probs.name)), y)
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
