from collections.abc import Sequence

import polars as pl
import pytest

from psycop.common.model_training_v2.classifier_pipelines.binary_classification_pipeline import (
    BinaryClassificationPipeline,
)
from psycop.common.model_training_v2.classifier_pipelines.estimator_steps.logistic_regression import (
    logistic_regression_step,
)
from psycop.common.model_training_v2.metrics.binary_metrics.base_binary_metric import (
    BinaryMetric,
)
from psycop.common.model_training_v2.metrics.binary_metrics.binary_auroc import (
    BinaryAUROC,
)
from psycop.common.model_training_v2.presplit_preprocessing.polars_frame import (
    PolarsFrame,
)
from psycop.common.model_training_v2.problem_type.binary_classification import (
    BinaryClassification,
)
from psycop.common.model_training_v2.problem_type.eval_dataset_base import (
    BinaryEvalDataset,
)
from psycop.common.model_training_v2.training_method.base_training_method import (
    TrainingResult,
)


@pytest.mark.parametrize(
    ("pipe", "main_metric", "supplementary_metrics", "x", "y", "main_metric_expected"),
    [
        (
            BinaryClassificationPipeline(steps=[logistic_regression_step()]),
            BinaryAUROC(),
            None,
            pl.DataFrame({"x": [1, 1, 2, 2]}),
            pl.Series("y", [0, 0, 1, 1]),
            1.0,
        ),
    ],
)
def test_binary_classification(
    pipe: BinaryClassificationPipeline,
    main_metric: BinaryMetric,
    supplementary_metrics: Sequence[BinaryMetric] | None,
    x: PolarsFrame,
    y: pl.Series,
    main_metric_expected: float,
):
    binary_classification_problem = BinaryClassification(
        pipe=pipe,
        main_metric=main_metric,
        supplementary_metrics=supplementary_metrics,
    )
    binary_classification_problem.train(x=x, y=y)

    result = binary_classification_problem.evaluate(x=x, y=y)
    assert isinstance(result, TrainingResult)
    assert result.main_metric.value == main_metric_expected
    assert isinstance(result.eval_dataset, BinaryEvalDataset)
