"""Generate performance by positive threshold.

E.g. if predicted probability is .4, and threshold is .5, resolve to 0.
"""

from typing import TYPE_CHECKING

from psycop.common.model_evaluation.binary.performance_by_ppr.performance_by_ppr import (
    days_from_first_positive_to_diagnosis,
    generate_performance_by_ppr_table,
    get_days_from_first_positive_to_diagnosis_from_df,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.common.test_utils.str_to_df import str_to_df

if TYPE_CHECKING:
    import pandas as pd


def test_generate_performance_by_threshold_table(subsampled_eval_dataset: EvalDataset):
    positive_rates = [0.3, 0.2, 0.1]

    output_table: pd.DataFrame = generate_performance_by_ppr_table(  # type: ignore
        eval_dataset=subsampled_eval_dataset, positive_rates=positive_rates
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
    assert output_table["prop with ≥1 true positive"].is_monotonic_decreasing


def test_time_from_flag_to_diag(synth_eval_dataset: EvalDataset):
    warning_days_half = days_from_first_positive_to_diagnosis(
        eval_dataset=synth_eval_dataset, positive_rate=0.5, aggregation_method="sum"
    )

    warning_days_two_thirds = days_from_first_positive_to_diagnosis(
        eval_dataset=synth_eval_dataset, positive_rate=0.75, aggregation_method="sum"
    )

    assert warning_days_half < warning_days_two_thirds


def test_time_from_flag_to_diag_from_df():
    df = str_to_df(
        """id,pred,y,pred_timestamps,outcome_timestamps,
        1,1,1,2020-01-01,2020-01-03, # 2 days
        1,1,1,2020-01-02,2020-01-03, # Ignored: Same patient but smaller distance
        2,1,1,2020-01-02,2020-01-03, # 1 day
        2,1,0,2020-01-02,NaN, # Ignored: Not true positive
        """
    )

    output = get_days_from_first_positive_to_diagnosis_from_df(df=df, aggregation_method="sum")

    assert output == 3
