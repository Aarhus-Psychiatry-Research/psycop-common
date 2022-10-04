"""Tests for all our visualisations.

Mainly tests that they run without errors.
"""
# pylint: disable=missing-function-docstring
from pathlib import Path

import altair as alt
import pandas as pd
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from psycopt2d.utils import positive_rate_to_pred_probs
from psycopt2d.visualization import plot_prob_over_time
from psycopt2d.visualization.base_charts import plot_bar_chart
from psycopt2d.visualization.feature_importance import plot_feature_importances
from psycopt2d.visualization.performance_over_time import (
    plot_auc_by_time_from_first_visit,
    plot_metric_by_time_until_diagnosis,
    plot_performance_by_calendar_time,
)
from psycopt2d.visualization.sens_over_time import (
    create_sensitivity_by_time_to_outcome_df,
    plot_sensitivity_by_time_to_outcome,
)


@pytest.fixture(scope="function")
def df():
    repo_path = Path(__file__).parent
    path = repo_path / "test_data" / "synth_eval_data.csv"
    df = pd.read_csv(path)

    # Convert all timestamp cols to datetime[64]ns
    for col in [col for col in df.columns if "timestamp" in col]:
        df[col] = pd.to_datetime(df[col])

    return df


def test_prob_over_time(df):
    alt.data_transformers.disable_max_rows()

    plot_prob_over_time(
        patient_id=df["dw_ek_borger"],
        timestamp=df["timestamp"],
        pred_prob=df["pred_prob"],
        outcome_timestamp=df["timestamp_t2d_diag"],
        label=df["label"],
        look_behind_distance=500,
    )


def test_get_sens_by_time_to_outcome_df(df):
    create_sensitivity_by_time_to_outcome_df(
        labels=df["label"],
        y_hat_probs=df["pred"],
        outcome_timestamps=df["timestamp_t2d_diag"],
        prediction_timestamps=df["timestamp"],
        pred_proba_threshold=0.5,
    )


def test_plot_bar_chart(df):
    plot_df = create_sensitivity_by_time_to_outcome_df(
        labels=df["label"],
        y_hat_probs=df["pred"],
        outcome_timestamps=df["timestamp_t2d_diag"],
        prediction_timestamps=df["timestamp"],
        pred_proba_threshold=0.5,
    )
    plot_bar_chart(
        x_values=plot_df["days_to_outcome_binned"],
        y_values=plot_df["sens"],
        x_title="Days to outcome",
        y_title="Sensitivity",
    )


def test_plot_performance_by_calendar_time(df):
    alt.data_transformers.disable_max_rows()

    plot_performance_by_calendar_time(
        labels=df["label"],
        y_hat=df["pred"],
        timestamps=df["timestamp"],
        bin_period="M",
        metric_fn=roc_auc_score,
        y_title="AUC",
    )


def test_plot_metric_until_diagnosis(df):
    alt.data_transformers.disable_max_rows()

    plot_metric_by_time_until_diagnosis(
        labels=df["label"],
        y_hat=df["pred"],
        diagnosis_timestamps=df["timestamp_t2d_diag"],
        prediction_timestamps=df["timestamp"],
        metric_fn=f1_score,
        y_title="F1",
    )


def test_plot_auc_time_from_first_visit(df):
    alt.data_transformers.disable_max_rows()

    plot_auc_by_time_from_first_visit(
        labels=df["label"],
        y_hat_probs=df["pred_prob"],
        first_visit_timestamps=df["timestamp_first_pred_time"],
        prediction_timestamps=df["timestamp"],
    )


def test_sens_by_time_to_outcome(df):
    positive_rate_thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]

    pred_proba_thresholds = positive_rate_to_pred_probs(
        pred_probs=df["pred_prob"],
        positive_rate_thresholds=positive_rate_thresholds,
    )

    plot_sensitivity_by_time_to_outcome(  # noqa
        labels=df["label"],
        y_hat_probs=df["pred_prob"],
        outcome_timestamps=df["timestamp_t2d_diag"],
        prediction_timestamps=df["timestamp"],
        pred_proba_thresholds=pred_proba_thresholds,
        bins=[0, 30, 182, 365, 730, 1825],
    )


def test_plot_feature_importances():
    feature_names = ["very long feature name right here yeah", "feat2", "feat3"]
    feature_importance = [0.6, 0.3, 0.1]

    plot_feature_importances(
        feature_names,
        feature_importances=feature_importance,
        top_n_feature_importances=3,
    )
