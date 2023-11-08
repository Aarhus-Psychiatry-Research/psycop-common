from collections.abc import Sequence

import pandas as pd
import polars as pl
import pytest

from psycop.common.model_training_v2.training_method.preprocessing.polars_frame import (
    PolarsFrame,
)
from psycop.common.model_training_v2.training_method.problem_type.binary_classification.binary_classification_pipeline import (
    BinaryClassificationPipeline,
)
from psycop.common.model_training_v2.training_method.problem_type.estimator_steps.logistic_regression import (
    logistic_regression_step,
)
from psycop.common.model_training_v2.training_method.problem_type.model_step import (
    ModelStep,
)


@pytest.mark.parametrize(
    ("steps", "x", "y"),
    [
        (
            [logistic_regression_step()],
            pl.DataFrame({"x": [1, 2, 3, 4]}),
            pl.Series([0, 0, 1, 1]),
        ),
    ],
)
def test_binary_classification_pipeline(
    steps: Sequence[ModelStep],
    x: PolarsFrame,
    y: pl.Series,
):
    pipeline = BinaryClassificationPipeline(steps=steps)
    pipeline.fit(x=x, y=y)

    y_hat_probs = pipeline.predict_proba(x=x)
    assert isinstance(y_hat_probs, pd.Series)
    assert y_hat_probs.name == "y_hat_probs"
