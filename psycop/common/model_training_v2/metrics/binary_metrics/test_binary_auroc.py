import polars as pl
import pytest

from psycop.common.model_training_v2.metrics.binary_metrics.binary_auroc import (
    BinaryAUROC,
)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        (
            pl.Series([1, 1, 0, 0]),
            pl.Series([0.9, 0.9, 0.1, 0.1]),
            1.0,
        ),
        (
            pl.Series([1, 0, 1, 0]),
            pl.Series([0.9, 0.9, 0.9, 0.9]),
            0.5,
        ),
    ],
)
def test_binary_auroc(y_true: pl.Series, y_pred: pl.Series, expected: float):
    auroc = BinaryAUROC()
    calculated_metric = auroc.calculate(y_true=y_true, y_pred=y_pred)
    assert calculated_metric.value == expected
