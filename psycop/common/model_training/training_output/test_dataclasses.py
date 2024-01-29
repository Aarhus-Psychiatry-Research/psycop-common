import numpy as np
import pandas as pd
import pytest

from psycop.common.model_training.training_output.dataclasses import (
    EvalDataset,
    get_predictions_for_positive_rate,
    get_predictions_for_threshold,
)


@pytest.mark.parametrize("desired_positive_rate", [0.1, 0.3, 0.5, 0.8])
def test_get_predictions_for_positive_rate(desired_positive_rate: float):
    df = pd.DataFrame({"y_hat_probs": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]})

    df["y_hat"] = get_predictions_for_positive_rate(
        desired_positive_rate=desired_positive_rate, y_hat_probs=df["y_hat_probs"]
    )[0]

    assert df["y_hat"].mean() == desired_positive_rate
    assert df["y_hat"].corr(df["y_hat_probs"]) > 0


@pytest.mark.parametrize("desired_threshold", [0.1, 0.5, 0.99])
def test_get_predictions_for_threshold(desired_threshold: float):
    df = pd.DataFrame({"y_hat_probs": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]})

    df["y_hat"] = get_predictions_for_threshold(
        desired_threshold=desired_threshold, y_hat_probs=df["y_hat_probs"]
    )[0]

    assert df["y_hat"].sum() == np.ceil((1 - desired_threshold) * 10)


def test_eval_dataset_direction_of_positives_for_quantile(subsampled_eval_dataset: EvalDataset):
    (positives_twenty_percent, _) = subsampled_eval_dataset.get_predictions_for_positive_rate(0.8)
    (positives_zero_percent, _) = subsampled_eval_dataset.get_predictions_for_positive_rate(0.0)

    assert positives_twenty_percent.mean() > positives_zero_percent.mean()
