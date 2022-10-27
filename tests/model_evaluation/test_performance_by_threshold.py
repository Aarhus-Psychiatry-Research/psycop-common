"""Generate performance by positive threshold.

E.g. if predicted probability is .4, and threshold is .5, resolve to 0.
"""

# pylint: disable=missing-function-docstring


import pandas as pd

from psycopt2d.evaluation_dataclasses import (
    ArtifactContainer,
    ArtifactSpecification,
    EvalDataset,
)
from psycopt2d.tables.performance_by_threshold import (
    days_from_first_positive_to_diagnosis,
    generate_performance_by_positive_rate_table,
)
from psycopt2d.utils.utils import positive_rate_to_pred_probs


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
            "threshold_percentile": {0: 90.0, 1: 50.0, 2: 10.0},
            "true_prevalence": {0: 0.0502, 1: 0.0502, 2: 0.0502},
            "positive_rate": {0: 0.1, 1: 0.5, 2: 0.5511},
            "negative_rate": {0: 0.9, 1: 0.5, 2: 0.4489},
            "PPV": {0: 0.0508, 1: 0.0502, 2: 0.0502},
            "NPV": {0: 0.9498, 1: 0.9497, 2: 0.9497},
            "sensitivity": {0: 0.1011, 1: 0.4997, 2: 0.5503},
            "specificity": {0: 0.9001, 1: 0.5, 2: 0.4488},
            "FPR": {0: 0.0999, 1: 0.5, 2: 0.5512},
            "FNR": {0: 0.8989, 1: 0.5003, 2: 0.4497},
            "accuracy": {0: 0.8599, 1: 0.5, 2: 0.4539},
            "true_positives": {0: 508, 1: 2510, 2: 2764},
            "true_negatives": {0: 85485, 1: 47487, 2: 42627},
            "false_positives": {0: 9492, 1: 47490, 2: 52350},
            "false_negatives": {0: 4515, 1: 2513, 2: 2259},
            "total_warning_days": {0: 609757.0, 1: 2619787.0, 2: 4612729.0},
            "warning_days_per_false_positive": {0: 64.2, 1: 55.2, 2: 88.1},
            "mean_warning_days": {0: 1252, 1: 1332, 2: 1451},
        },
    )

    for col in output_table.columns:
        output_table[col].equals(expected_df[col])


def test_time_from_flag_to_diag(synth_eval_dataset: EvalDataset):
    # Threshold = 0.5
    val = days_from_first_positive_to_diagnosis(
        ids=synth_eval_dataset.ids,
        pred_probs=synth_eval_dataset.y_hat_probs,
        pred_timestamps=synth_eval_dataset.pred_timestamps,
        outcome_timestamps=synth_eval_dataset.outcome_timestamps,
        positive_rate_threshold=0.5,
    )

    assert 290_000 < val < 292_000

    # Threshold = 0.2
    val = days_from_first_positive_to_diagnosis(
        ids=synth_eval_dataset.ids,
        pred_probs=synth_eval_dataset.y_hat_probs,
        pred_timestamps=synth_eval_dataset.pred_timestamps,
        outcome_timestamps=synth_eval_dataset.outcome_timestamps,
        positive_rate_threshold=0.2,
    )

    assert 1_875_000 < val < 1_885_000
