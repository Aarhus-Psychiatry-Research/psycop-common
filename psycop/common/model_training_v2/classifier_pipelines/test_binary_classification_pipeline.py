from collections.abc import Sequence

import pandas as pd
import polars as pl
import pytest

from psycop.common.model_training_v2.classifier_pipelines.binary_classification_pipeline import (
    BinaryClassificationPipeline,
)
from psycop.common.model_training_v2.classifier_pipelines.estimator_steps.xgboost import (
    xgboost_classifier_step,
)
from psycop.common.model_training_v2.classifier_pipelines.model_step import ModelStep
from psycop.common.model_training_v2.presplit_preprocessing.polars_frame import (
    PolarsFrame,
)


@pytest.mark.parametrize(
    ("steps", "x", "y"),
    [
        (
            [xgboost_classifier_step()],
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
