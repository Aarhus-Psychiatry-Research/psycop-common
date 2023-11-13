import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from sklearn.pipeline import Pipeline

from psycop.common.model_training_v2.trainer.task.binary_classification.binary_classification import (
    BinaryClassification,
)
from psycop.common.model_training_v2.trainer.task.binary_classification.binary_classification_pipeline import (
    BinaryClassificationPipeline,
)
from psycop.common.model_training_v2.trainer.task.binary_classification.binary_metrics.base_binary_metric import (
    BinaryMetric,
)
from psycop.common.model_training_v2.trainer.task.binary_classification.binary_metrics.binary_auroc import (
    BinaryAUROC,
)
from psycop.common.model_training_v2.trainer.task.estimator_steps.logistic_regression import (
    logistic_regression_step,
)


@pytest.mark.parametrize(
    ("pipe", "main_metric", "x", "y", "main_metric_expected"),
    [
        (
            BinaryClassificationPipeline(
                sklearn_pipe=Pipeline([logistic_regression_step()]),
            ),
            BinaryAUROC(),
            pd.DataFrame({"x": [1, 1, 2, 2], "uuid": [1, 2, 3, 4]}),
            pd.DataFrame({"y": [0, 0, 1, 1]}),
            1.0,
        ),
    ],
)
def test_binary_classification(
    pipe: BinaryClassificationPipeline,
    main_metric: BinaryMetric,
    x: pd.DataFrame,
    y: pd.DataFrame,
    main_metric_expected: float,
):
    binary_classification_problem = BinaryClassification(
        task_pipe=pipe,
        main_metric=main_metric,
        pred_time_uuid_col_name="uuid",
    )
    binary_classification_problem.train(x=x, y=y, y_col_name="y")

    result = binary_classification_problem.evaluate(x=x, y=y, y_col_name="y")
    assert result.metric.value == main_metric_expected

    pred_uuids = result.eval_dataset.df.to_pandas()[result.eval_dataset.pred_time_uuids]
    assert_series_equal(pred_uuids, x["uuid"])
