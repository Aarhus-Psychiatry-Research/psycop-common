"""Generate performance by positive threshold.

E.g. if predicted probability is .4, and threshold is .5, resolve to 0.
"""

# pylint: disable=missing-function-docstring


import pandas as pd
from psycop_model_training.model_eval.base_artifacts.tables.performance_by_threshold import (
    days_from_first_positive_to_diagnosis,
    generate_performance_by_positive_rate_table,
)
from psycop_model_training.model_eval.dataclasses import ArtifactContainer, EvalDataset
from psycop_model_training.utils.utils import positive_rate_to_pred_probs


def test_generate_performance_by_threshold_table(synth_eval_dataset: EvalDataset):
    positive_rate_thresholds = [0.9, 0.5, 0.1]

    pred_proba_thresholds = positive_rate_to_pred_probs(
        pred_probs=synth_eval_dataset.y_hat_probs,
        positive_rate_thresholds=positive_rate_thresholds,
    )

    table_spec = ArtifactContainer(
        label="performance_by_threshold_table",
        artifact=generate_performance_by_positive_rate_table(
            eval_dataset=synth_eval_dataset,
            positive_rate_thresholds=positive_rate_thresholds,
            pred_proba_thresholds=pred_proba_thresholds,
            output_format="df",
        ),
    )

    output_table: pd.DataFrame = table_spec.artifact

    assert output_table["true_prevalence"].std() == 0
    assert output_table["positive_rate"].is_monotonic_decreasing
    assert output_table["negative_rate"].is_monotonic_increasing
    assert output_table["PPV"].std() < 0.01
    assert output_table["NPV"].std() < 0.01
    assert output_table["sensitivity"].is_monotonic_decreasing
    assert output_table["specificity"].is_monotonic_increasing
    assert output_table["FPR"].is_monotonic_decreasing
    assert output_table["FNR"].is_monotonic_increasing
    assert output_table["accuracy"].is_monotonic_increasing
    assert output_table["true_positives"].is_monotonic_decreasing
    assert output_table["true_negatives"].is_monotonic_increasing
    assert output_table["false_positives"].is_monotonic_decreasing
    assert output_table["false_negatives"].is_monotonic_increasing
    assert output_table["total_warning_days"].is_monotonic_decreasing
    assert output_table["warning_days_per_false_positive"].dtype == "float64"
    assert output_table["mean_warning_days"].is_monotonic_decreasing
    assert output_table["prop_with_at_least_one_true_positive"].is_monotonic_decreasing


def test_time_from_flag_to_diag(synth_eval_dataset: EvalDataset):
    # Threshold is 0.5
    val = days_from_first_positive_to_diagnosis(
        eval_dataset=synth_eval_dataset,
        positive_rate_threshold=0.5,
    )

    assert 260_000 < val < 292_000

    # Threshold is 0.2
    val = days_from_first_positive_to_diagnosis(
        eval_dataset=synth_eval_dataset,
        positive_rate_threshold=0.2,
    )

    assert 1_800_000 < val < 1_885_000
