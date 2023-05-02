"""Tests for all our visualisations.

Mainly tests that they run without errors.
"""


from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from psycop.model_evaluation.base_charts import (
    plot_basic_chart,
)
from psycop.model_evaluation.binary.global_performance.precision_recall import (
    plot_precision_recall,
)
from psycop.model_evaluation.binary.global_performance.roc_auc import plot_auc_roc
from psycop.model_evaluation.binary.subgroups.age import plot_roc_auc_by_age
from psycop.model_evaluation.binary.subgroups.sex import plot_roc_auc_by_sex
from psycop.model_evaluation.binary.time.absolute_plots import (
    plot_metric_by_absolute_time,
    plot_prob_over_time,
)
from psycop.model_evaluation.binary.time.periodic_plots import (
    plot_roc_auc_by_periodic_time,
)
from psycop.model_evaluation.binary.time.timedelta_data import (
    create_sensitivity_by_time_to_outcome_df,
)
from psycop.model_evaluation.binary.time.timedelta_plots import (
    plot_roc_auc_by_time_from_first_visit,
    plot_sensitivity_by_time_to_event,
    plot_sensitivity_by_time_until_diagnosis,
    plot_time_from_first_positive_to_event,
)
from psycop.model_evaluation.feature_importance.sklearn.feature_importance import (
    plot_feature_importances,
)
from psycop.model_evaluation.utils import TEST_PLOT_PATH
from psycop.model_training.training_output.dataclasses import EvalDataset


def test_prob_over_time(subsampled_eval_dataset: EvalDataset, tmp_path: str):
    plot_prob_over_time(
        patient_id=subsampled_eval_dataset.ids,
        timestamp=subsampled_eval_dataset.pred_timestamps,
        pred_prob=subsampled_eval_dataset.y_hat_probs,
        outcome_timestamp=subsampled_eval_dataset.outcome_timestamps,
        label=subsampled_eval_dataset.y,
        look_behind_distance=500,
        save_path=Path(tmp_path),
    )


def test_get_sens_by_time_to_outcome_df(synth_eval_dataset: EvalDataset):
    create_sensitivity_by_time_to_outcome_df(
        eval_dataset=synth_eval_dataset,
        outcome_timestamps=synth_eval_dataset.outcome_timestamps,
        prediction_timestamps=synth_eval_dataset.pred_timestamps,
        desired_positive_rate=0.5,
    )


def test_plot_bar_chart(synth_eval_dataset: EvalDataset):
    plot_df = create_sensitivity_by_time_to_outcome_df(
        eval_dataset=synth_eval_dataset,
        outcome_timestamps=synth_eval_dataset.outcome_timestamps,
        prediction_timestamps=synth_eval_dataset.pred_timestamps,
        desired_positive_rate=0.5,
    )
    plot_basic_chart(
        x_values=plot_df["days_to_outcome_binned"],  # type: ignore
        y_values=plot_df["sens"],  # type: ignore
        x_title="Days to outcome",
        y_title="Sensitivity",
        plot_type="bar",
        save_path=TEST_PLOT_PATH / "test_plot_basic_chart.png",
    )


def test_plot_performance_by_age(subsampled_eval_dataset: EvalDataset):
    plot_roc_auc_by_age(
        eval_dataset=subsampled_eval_dataset,
        save_path=TEST_PLOT_PATH / "test_performance_plot_by_age.png",
    )


def test_plot_performance_by_sex(subsampled_eval_dataset: EvalDataset):
    plot_roc_auc_by_sex(
        eval_dataset=subsampled_eval_dataset,
        save_path=TEST_PLOT_PATH / "test_performance_plot_by_sex.png",
    )


@pytest.mark.parametrize(
    "bin_period",
    ["Q", "Y"],
)
def test_plot_performance_by_calendar_time(
    subsampled_eval_dataset: EvalDataset,
    bin_period: Literal["M", "Q", "Y"],
):
    plot_metric_by_absolute_time(
        eval_dataset=subsampled_eval_dataset,
        bin_period=bin_period,
        confidence_interval=0.95,
        save_path=TEST_PLOT_PATH / f"test_{bin_period}.png",  # type: ignore
    )


def test_sensitivity_by_time_to_event(
    subsampled_eval_dataset: EvalDataset,
):
    # TODO: Another one that is currently failing, the next person who uses it should debug it
    plot_sensitivity_by_time_to_event(
        eval_dataset=subsampled_eval_dataset,
        positive_rates=[0.4, 0.6, 0.8],
        bins=list(range(0, 1460, 180)),
        n_bootstraps=10,
        y_limits=(0, 1),
        save_path=TEST_PLOT_PATH / "sensitivity_by_time_to_event.png",
    )


@pytest.mark.parametrize(
    "bin_period",
    ["H", "D", "M"],
)
def test_plot_performance_by_cyclic_time(
    subsampled_eval_dataset: EvalDataset,
    bin_period: str,
):
    plot_roc_auc_by_periodic_time(
        eval_dataset=subsampled_eval_dataset,
        bin_period=bin_period,
    )


@pytest.mark.xfail()
def test_plot_metric_until_diagnosis(subsampled_eval_dataset: EvalDataset):
    # TODO: Another one that is currently failing, the next person who uses it should debug it
    plot_sensitivity_by_time_until_diagnosis(
        eval_dataset=subsampled_eval_dataset,
        y_title="Sensitivity (recall)",
        confidence_interval=0.95,
        save_path=TEST_PLOT_PATH / "sensitivity_by_time_until_diagnosis.png",
    )

    # TODO: This is failing for no known reason. The next person who needs it should debug it.


def test_plot_auc_time_from_first_visit(subsampled_eval_dataset: EvalDataset):
    plot_roc_auc_by_time_from_first_visit(
        eval_dataset=subsampled_eval_dataset,
    )


def test_plot_feature_importances():
    n_features = 10
    feature_name = "very long feature name right here yeah actually super long like the feature names"
    feature_names = [feature_name + str(i) for i in range(n_features)]
    # generate 10 random numbers between 0 and 1
    feature_importance = np.random.rand(n_features)

    feature_importance_dict = dict(zip(feature_names, feature_importance))

    plot_feature_importances(
        feature_importance_dict=feature_importance_dict,
        top_n_feature_importances=n_features,
        save_path=TEST_PLOT_PATH / "tmp",
    )


def test_plot_roc_auc(subsampled_eval_dataset: EvalDataset):
    plot_auc_roc(
        eval_dataset=subsampled_eval_dataset,
        save_path=TEST_PLOT_PATH / "roc_auc.png",
        n_bootstraps=10,
    )


# @pytest.mark.skip(reason="Breaking on ubuntu only, don't have time to debug right now")
def test_plot_time_from_first_positive_to_event(subsampled_eval_dataset: EvalDataset):
    plot_time_from_first_positive_to_event(
        eval_dataset=subsampled_eval_dataset,
        bins=list(range(0, 60, 3)),
        min_n_in_bin=1,
        save_path=TEST_PLOT_PATH / "time_from_first_positive_to_event.png",
    )


def test_plot_precision_recall(subsampled_eval_dataset: EvalDataset):
    plot_precision_recall(
        eval_dataset=subsampled_eval_dataset,
        save_path=Path(
            TEST_PLOT_PATH / "precision_recall.png",
        ),
    )


@pytest.mark.xfail()
def test_overlay_barplot(subsampled_eval_dataset: EvalDataset):
    plot_sensitivity_by_time_until_diagnosis(
        eval_dataset=subsampled_eval_dataset,
        y_title="Sensitivity",
        confidence_interval=0.95,
        save_path=TEST_PLOT_PATH / "test_overlay_barplot.png",
    )

    # TODO: This is failing for no known reason. The next person who needs it should debug it.
