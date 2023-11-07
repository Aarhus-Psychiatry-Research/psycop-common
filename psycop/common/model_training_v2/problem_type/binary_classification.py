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
from psycop.common.model_training_v2.problem_type.problem_type_base import ProblemType
from psycop.common.model_training_v2.training_method.base_training_method import (
    TrainingResult,
)


class BinaryClassification:
    def __init__(
        self,
        pipe: BinaryClassificationPipeline,
        main_metric: BinaryMetric,
    ):
        self.pipe = pipe
        self.main_metric = main_metric

    def train(
        self,
        x: PolarsFrame,
        y: pl.Series,
    ) -> None:
        self.pipe.fit(x=x, y=y)
        self.is_fitted = True

    def predict_proba(self, x: PolarsFrame) -> PredProbaSeries:
        return self.pipe.predict_proba(x)

    def evaluate(self, x: PolarsFrame, y: pl.Series) -> TrainingResult:
        if isinstance(x, pl.LazyFrame):
            x = x.collect()
        y_hat_probs = self.pipe.predict_proba(x)

        df = x.with_columns(pl.Series(y_hat_probs).alias(str(y_hat_probs.name)), y)

        eval_dataset = BinaryEvalDataset(
            pred_time_uuids="pred_time_uuids",  # TODO: #383 Ensure that pred_time_uuids are passed through the entire pipeline
            y_hat_probs=str(y_hat_probs.name),
            y=y.name,
            df=df,
        )
        main_metric = eval_dataset.calculate_metrics([self.main_metric])[0]

        return TrainingResult(
            metric=main_metric,
            eval_dataset=eval_dataset,
        )
