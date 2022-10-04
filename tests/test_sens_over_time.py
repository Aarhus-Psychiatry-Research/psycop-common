"""Tests of sens over time."""
# pylint: disable=missing-function-docstring
from pathlib import Path

import pandas as pd
import pytest

# from psycopt2d.utils import positive_rate_to_pred_probs
# from psycopt2d.visualization.sens_over_time import plot_sensitivity_by_time_to_outcome


@pytest.fixture(scope="function")
def df():
    repo_path = Path(__file__).parent
    path = repo_path / "test_data" / "synth_eval_data.csv"
    df = pd.read_csv(path)

    # Convert all timestamp cols to datetime[64]ns
    for col in [col for col in df.columns if "timestamp" in col]:
        df[col] = pd.to_datetime(df[col])

    return df


# def test_plot_sensitivity_by_time_to_outcome(df):
# Disabled because of errors with altair. We're migrating away, re-enable when we're done.

# positive_rates = [0.95, 0.99, 0.999, 0.9999]

# pred_proba_thresholds = positive_rate_to_pred_probs(
#     pred_probs=df["pred_prob"],
#     positive_rate_thresholds=positive_rates,
# )

# plt = plot_sensitivity_by_time_to_outcome(
#     labels=df["label"],
#     y_hat_probs=df["pred_prob"],
#     pred_proba_thresholds=pred_proba_thresholds,
#     outcome_timestamps=df["timestamp_t2d_diag"],
#     prediction_timestamps=df["timestamp"],
#     bins=[0, 28, 182, 365, 730, 1825],
# )

# plt.save("test_plot_sensitivity_by_time_to_outcome.png")
# pass
