import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from psycop.common.model_training_v2.trainer.task.binary_classification.binary_classification_pipeline import (
    BinaryClassificationPipeline,
)
from psycop.common.model_training_v2.trainer.task.estimator_steps.logistic_regression import (
    logistic_regression_step,
)


@pytest.mark.parametrize(
    ("pipe", "x", "y"),
    [
        (
            Pipeline([logistic_regression_step()]),
            pd.DataFrame({"x": [1, 2, 3, 4]}),
            pd.Series([0, 0, 1, 1]),
        ),
    ],
)
def test_binary_classification_pipeline(
    pipe: Pipeline,
    x: pd.DataFrame,
    y: pd.Series, # type: ignore
):
    pipeline = BinaryClassificationPipeline(sklearn_pipe=pipe)
    pipeline.fit(x=x, y=y)

    y_hat_probs = pipeline.predict_proba(x=x)
    assert isinstance(y_hat_probs, pd.Series)
    assert y_hat_probs.name == "y_hat_probs"
