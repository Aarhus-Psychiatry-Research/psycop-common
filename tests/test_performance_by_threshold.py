"""Generate performance by positive threshold.

E.g. if predicted probability is .4, and threshold is .5, resolve to 0.
"""


import pandas as pd
from psycop_model_evaluation.binary.performance_by_true_positive_rate import (
    days_from_first_positive_to_diagnosis,
    generate_performance_by_positive_rate_table,
)
from psycop_model_training.training_output.dataclasses import EvalDataset


def test_generate_performance_by_threshold_table(synth_eval_dataset: EvalDataset):
    positive_rates = [0.9, 0.5, 0.1]

    output_table: pd.DataFrame = generate_performance_by_positive_rate_table(  # type: ignore
        eval_dataset=synth_eval_dataset,
        positive_rates=positive_rates,
        output_format="df",
    )
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
    warning_days_half = days_from_first_positive_to_diagnosis(
        eval_dataset=synth_eval_dataset,
        positive_rate=0.5,
        aggregation_method="sum",
    )

    warning_days_two_thirds = days_from_first_positive_to_diagnosis(
        eval_dataset=synth_eval_dataset,
        positive_rate=0.75,
        aggregation_method="sum",
    )

    assert warning_days_half < warning_days_two_thirds
