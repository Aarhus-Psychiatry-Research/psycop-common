import polars as pl
import pytest

from psycop.common.model_training_v2.training_method.preprocessing.polars_frame import (
    PolarsFrame,
)
from psycop.common.model_training_v2.training_method.problem_type.binary_classification.binary_classification import (
    BinaryClassification,
)
from psycop.common.model_training_v2.training_method.problem_type.binary_classification.binary_classification_pipeline import (
    BinaryClassificationPipeline,
)
from psycop.common.model_training_v2.training_method.problem_type.binary_classification.binary_metrics.base_binary_metric import (
    BinaryMetric,
)
from psycop.common.model_training_v2.training_method.problem_type.binary_classification.binary_metrics.binary_auroc import (
    BinaryAUROC,
)
from psycop.common.model_training_v2.training_method.problem_type.estimator_steps.logistic_regression import (
    logistic_regression_step,
)


@pytest.mark.parametrize(
    ("pipe", "main_metric", "x", "y", "main_metric_expected"),
    [
        (
            BinaryClassificationPipeline(steps=[logistic_regression_step()]),
            BinaryAUROC(),
            pl.DataFrame({"x": [1, 1, 2, 2]}),
            pl.DataFrame({"y": [0, 0, 1, 1]}),
            1.0,
        ),
    ],
)
def test_binary_classification(
    pipe: BinaryClassificationPipeline,
    main_metric: BinaryMetric,
    x: PolarsFrame,
    y: pl.DataFrame,
    main_metric_expected: float,
):
    binary_classification_problem = BinaryClassification(
        pipe=pipe,
        main_metric=main_metric,
    )
    binary_classification_problem.train(x=x, y=y)

    result = binary_classification_problem.evaluate(x=x, y=y)
    assert result.metric.value == main_metric_expected
