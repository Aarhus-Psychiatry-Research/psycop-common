"""Tests for all our visualisations.

Mainly tests that they run without errors.
"""
# pylint: disable=missing-function-docstring
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from psycop_model_training.model_eval.base_artifacts.plots.base_charts import (
    plot_basic_chart,
)
from psycop_model_training.model_eval.base_artifacts.plots.feature_importance import (
    plot_feature_importances,
)
from psycop_model_training.model_eval.base_artifacts.plots.performance_by_age import (
    plot_performance_by_age,
)
from psycop_model_training.model_eval.base_artifacts.plots.performance_over_time import (
    plot_auc_by_time_from_first_visit,
    plot_metric_by_calendar_time,
    plot_metric_by_cyclic_time,
    plot_metric_by_time_until_diagnosis,
)
from psycop_model_training.model_eval.base_artifacts.plots.prob_over_time import (
    plot_prob_over_time,
)
from psycop_model_training.model_eval.base_artifacts.plots.roc_auc import plot_auc_roc
from psycop_model_training.model_eval.base_artifacts.plots.sens_over_time import (
    create_sensitivity_by_time_to_outcome_df,
    plot_sensitivity_by_time_to_outcome_heatmap,
)
from psycop_model_training.model_eval.dataclasses import EvalDataset
from psycop_model_training.utils.utils import PROJECT_ROOT, positive_rate_to_pred_probs


@pytest.fixture(scope="function")
def df():
    repo_path = Path(__file__).parent
    path = repo_path / "test_data" / "synth_eval_data.csv"
    df = pd.read_csv(path)

    # Convert all timestamp cols to datetime[64]ns
    for col in [col for col in df.columns if "timestamp" in col]:
        df[col] = pd.to_datetime(df[col])

    df["n_hba1c"] = np.random.randint(0, 8, df.shape[0])

    df["age"] = np.random.uniform(18, 90, df.shape[0])

    return df


def test_prob_over_time(synth_eval_dataset: EvalDataset, tmp_path):
    plot_prob_over_time(
        patient_id=synth_eval_dataset.ids,
        timestamp=synth_eval_dataset.pred_timestamps,
        pred_prob=synth_eval_dataset.y_hat_probs,
        outcome_timestamp=synth_eval_dataset.outcome_timestamps,
        label=synth_eval_dataset.y,
        look_behind_distance=500,
        save_path=tmp_path,
    )


def test_get_sens_by_time_to_outcome_df(synth_eval_dataset: EvalDataset):
    create_sensitivity_by_time_to_outcome_df(
        labels=synth_eval_dataset.y,
        y_hat_probs=synth_eval_dataset.y_hat_probs,
        outcome_timestamps=synth_eval_dataset.outcome_timestamps,
        prediction_timestamps=synth_eval_dataset.pred_timestamps,
        pred_proba_threshold=0.5,
    )


def test_plot_bar_chart(synth_eval_dataset: EvalDataset):
    plot_df = create_sensitivity_by_time_to_outcome_df(
        labels=synth_eval_dataset.y,
        y_hat_probs=synth_eval_dataset.y_hat_probs,
        outcome_timestamps=synth_eval_dataset.outcome_timestamps,
        prediction_timestamps=synth_eval_dataset.pred_timestamps,
        pred_proba_threshold=0.5,
    )
    plot_basic_chart(
        x_values=plot_df["days_to_outcome_binned"],
        y_values=plot_df["sens"],
        x_title="Days to outcome",
        y_title="Sensitivity",
        plot_type="bar",
        save_path=PROJECT_ROOT / "12345.png",
    )


def test_plot_performance_by_age(synth_eval_dataset: EvalDataset):
    plot_performance_by_age(
        eval_dataset=synth_eval_dataset,
        save_path=PROJECT_ROOT / "test.png",
    )


@pytest.mark.parametrize(
    "bin_period",
    ["M", "Q", "Y"],
)
def test_plot_performance_by_calendar_time(
    synth_eval_dataset: EvalDataset,
    bin_period: str,
):
    plot_metric_by_calendar_time(
        eval_dataset=synth_eval_dataset,
        bin_period=bin_period,
        metric_fn=roc_auc_score,
    )


@pytest.mark.parametrize(
    "bin_period",
    ["H", "D", "M"],
)
def test_plot_performance_by_cyclic_time(
    synth_eval_dataset: EvalDataset,
    bin_period: str,
):
    plot_metric_by_cyclic_time(
        eval_dataset=synth_eval_dataset,
        bin_period=bin_period,
        metric_fn=roc_auc_score,
    )


def test_plot_metric_until_diagnosis(synth_eval_dataset: EvalDataset):
    plot_metric_by_time_until_diagnosis(
        eval_dataset=synth_eval_dataset,
        metric_fn=f1_score,
        y_title="F1",
    )


def test_plot_auc_time_from_first_visit(synth_eval_dataset: EvalDataset):
    plot_auc_by_time_from_first_visit(
        eval_dataset=synth_eval_dataset,
    )


def test_plot_sens_by_time_to_outcome(synth_eval_dataset: EvalDataset, tmp_path):
    positive_rate_thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    pred_proba_thresholds = positive_rate_to_pred_probs(
        pred_probs=synth_eval_dataset.y_hat_probs,
        positive_rate_thresholds=positive_rate_thresholds,
    )

    plot_sensitivity_by_time_to_outcome_heatmap(
        eval_dataset=synth_eval_dataset,
        pred_proba_thresholds=pred_proba_thresholds,
        bins=[0, 30, 182, 365, 730, 1825],
        save_path=tmp_path,
    )


def test_plot_feature_importances():
    n_features = 10
    feature_name = "very long feature name right here yeah actually super long like the feature names"
    feature_names = [feature_name + str(i) for i in range(n_features)]
    # generate 10 random nubmers between 0 and 1
    feature_importance = np.random.rand(n_features)

    feature_importance_dict = dict(zip(feature_names, feature_importance))

    plot_feature_importances(
        feature_importance_dict=feature_importance_dict,
        top_n_feature_importances=n_features,
        save_path="tmp",
    )


def test_plot_roc_auc(synth_eval_dataset: EvalDataset):
    plot_auc_roc(eval_dataset=synth_eval_dataset)
