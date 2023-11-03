import polars as pl
import pytest

from psycop.common.model_training_v2.metrics.binary_metrics.binary_auroc import (
    BinaryAUROC,
)
from psycop.common.model_training_v2.presplit_preprocessing.polars_frame import (
    PolarsFrame,
)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        (pl.Series([0, 1, 1, 0]), pl.Series([0.1, 0.2, 0.3, 0.4]), 0.5),
        (pl.Series([0, 1, 1, 0]), pl.Series([0.1, 0.9, 0.9, 0.1]), 1.0),
    ],
)
def test_binary_auroc(y_true: PolarsFrame, y_pred: pl.Series, expected: float):
    metric = BinaryAUROC()
    assert metric(y_true, y_pred) == expected
