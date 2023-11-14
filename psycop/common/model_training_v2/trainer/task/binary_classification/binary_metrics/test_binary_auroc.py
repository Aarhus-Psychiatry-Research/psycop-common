from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from psycop.common.model_training_v2.trainer.task.binary_classification.binary_metrics.binary_auroc import (
    BinaryAUROC,
)

if TYPE_CHECKING:
    from psycop.common.model_training_v2.trainer.task.base_metric import (
        PredProbaSeries,
    )


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        (
            pd.Series([1, 1, 0, 0]),
            pd.Series([0.9, 0.9, 0.1, 0.1]),
            1.0,
        ),
        (
            pd.Series([1, 0, 1, 0]),
            pd.Series([0.9, 0.9, 0.9, 0.9]),
            0.5,
        ),
    ],
)
def test_binary_auroc(y_true: pd.Series[int], y_pred: PredProbaSeries, expected: float):
    auroc = BinaryAUROC()
    calculated_metric = auroc.calculate(y=y_true, y_hat_prob=y_pred)
    assert calculated_metric.value == expected
