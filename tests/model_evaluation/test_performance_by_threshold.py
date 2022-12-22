"""Generate performance by positive threshold.

E.g. if predicted probability is .4, and threshold is .5, resolve to 0.
"""

# pylint: disable=missing-function-docstring


import pandas as pd

from psycop_model_training.model_eval.artifacts.tables import (
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

    output_table = table_spec.artifact

    expected_df = pd.DataFrame(
        {
            "true_prevalence": [
                0.0502,
                0.0502,
                0.0502,
            ],
            "positive_rate": [
                0.5511,
                0.5,
                0.1,
            ],
            "negative_rate": [
                0.4489,
                0.5,
                0.9,
            ],
            "PPV": [
                0.0502,
                0.0502,
                0.0508,
            ],
            "NPV": [
                0.9497,
                0.9497,
                0.9498,
            ],
            "sensitivity": [
                0.5503,
                0.4997,
                0.1011,
            ],
            "specificity": [
                0.4488,
                0.5,
                0.9001,
            ],
            "FPR": [
                0.5512,
                0.5,
                0.0999,
            ],
            "FNR": [
                0.4497,
                0.5003,
                0.8989,
            ],
            "accuracy": [
                0.4539,
                0.5,
                0.8599,
            ],
            "true_positives": [
                2764,
                2510,
                508,
            ],
            "true_negatives": [
                42627,
                47487,
                85485,
            ],
            "false_positives": [
                52350,
                47490,
                9492,
            ],
            "false_negatives": [
                2259,
                2513,
                4515,
            ],
            "total_warning_days": [
                4612729.0,
                2619787.0,
                609757.0,
            ],
            "warning_days_per_false_positive": [
                88.1,
                55.2,
                64.2,
            ],
            "mean_warning_days": [
                1451.0,
                1332.0,
                1252.0,
            ],
            "prop_with_at_least_one_true_positive": [
                0.0503,
                0.0311,
                0.0077,
            ],
        },
    )

    for col in output_table.columns:
        assert output_table[col].equals(expected_df[col])


def test_time_from_flag_to_diag(synth_eval_dataset: EvalDataset):
    # Threshold = 0.5
    val = days_from_first_positive_to_diagnosis(
        eval_dataset=synth_eval_dataset,
        positive_rate_threshold=0.5,
    )

    assert 290_000 < val < 292_000

    # Threshold = 0.2
    val = days_from_first_positive_to_diagnosis(
        eval_dataset=synth_eval_dataset,
        positive_rate_threshold=0.2,
    )

    assert 1_875_000 < val < 1_885_000
