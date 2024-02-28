"""Get performance by which threshold is used to classify positive. PPR means predicted positive rate, i.e. the proportion of the population that is predicted to be positive."""
from collections.abc import Sequence

import numpy as np
import pandas as pd

from psycop.common.model_evaluation.binary.performance_by_ppr.prop_of_all_events_hit_by_true_positive import (
    get_prop_of_events_captured_from_eval_dataset,
)
from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_time_from_first_positive_to_diagnosis_df,
)
from psycop.common.model_evaluation.confusion_matrix.confusion_matrix import (
    get_confusion_matrix_cells_from_df,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset


def get_true_positives(eval_dataset: EvalDataset, positive_rate: float = 0.5) -> pd.DataFrame:
    """Get dataframe containing only true positives.

    Args:
        eval_dataset (EvalDataset): EvalDataset object.
        positive_rate (float): Takes the top positive_rate% of predicted probabilities and turns them into 1, the rest 0.

    Returns:
        pd.DataFrame: Dataframe containing only true positives.
    """

    # Generate df
    positives_series, _ = eval_dataset.get_predictions_for_positive_rate(
        desired_positive_rate=positive_rate
    )

    df = pd.DataFrame(
        {
            "id": eval_dataset.ids,
            "pred": positives_series,
            "y": eval_dataset.y,
            "pred_timestamps": eval_dataset.pred_timestamps,
            "outcome_timestamps": eval_dataset.outcome_timestamps,
        }
    )

    # Keep only true positives
    df["true_positive"] = (df["pred"] == 1) & (df["y"] == 1)

    return df[df["true_positive"]]


def performance_by_ppr(eval_dataset: EvalDataset, positive_rate: float) -> pd.DataFrame:
    """Generates a row for a performance_by_threshold table.

    Args:
        eval_dataset (EvalDataset): EvalDataset object.
        positive_rate (float): Takes the top positive_rate% of predicted probabilities and turns them into 1, the rest 0.
        round_to (int): Number of decimal places to round metrics

    Returns:
        pd.DataFrame
    """
    preds, _ = eval_dataset.get_predictions_for_positive_rate(positive_rate)

    conf_matrix_df = pd.DataFrame({"pred": preds, "true": eval_dataset.y})

    conf_matrix = get_confusion_matrix_cells_from_df(df=conf_matrix_df)

    true_neg = conf_matrix.true_negatives
    false_neg = conf_matrix.false_negatives
    true_pos = conf_matrix.true_positives
    false_pos = conf_matrix.false_positives

    n_total = true_neg + false_neg + true_pos + false_pos

    true_prevalence = (true_pos + false_neg) / n_total

    positive_rate = (true_pos + false_pos) / n_total
    negative_rate = (true_neg + false_neg) / n_total

    pos_pred_val = true_pos / (true_pos + false_pos)
    neg_pred_val = true_neg / (true_neg + false_neg)

    sens = true_pos / (true_pos + false_neg)
    spec = true_neg / (true_neg + false_pos)

    false_pos_rate = false_pos / (true_neg + false_pos)
    neg_pos_rate = false_neg / (true_pos + false_neg)

    acc = (true_pos + true_neg) / n_total

    precision = pos_pred_val
    recall = sens
    f1 = 2 * (precision * recall) / (precision + recall)
    mcc = (true_pos * true_neg - false_pos * false_neg) / np.sqrt(
        (true_pos + false_pos)
        * (true_pos + false_neg)
        * (true_neg + false_pos)
        * (true_neg + false_neg)
    )

    # Must return lists as values, otherwise pd.Dataframe requires setting indices
    metrics_matrix = pd.DataFrame(
        {
            "positive_rate": [positive_rate],
            "negative_rate": [negative_rate],
            "true_prevalence": [true_prevalence],
            "PPV": [pos_pred_val],
            "NPV": [neg_pred_val],
            "sensitivity": [sens],
            "specificity": [spec],
            "FPR": [false_pos_rate],
            "FNR": [neg_pos_rate],
            "accuracy": [acc],
            "true_positives": [true_pos],
            "true_negatives": [true_neg],
            "false_positives": [false_pos],
            "false_negatives": [false_neg],
            "f1": [f1],
            "mcc": [mcc],
        }
    )

    return metrics_matrix.reset_index(drop=True)


def days_from_first_positive_to_diagnosis(
    eval_dataset: EvalDataset, positive_rate: float = 0.5, aggregation_method: str = "sum"
) -> float:
    """Calculate number of days from the first positive prediction to the
    patient's outcome timestamp.

    Args:
        eval_dataset (EvalDataset): EvalDataset object.
        positive_rate (float): Takes the top positive_rate% of predicted probabilities and turns them into 1, the rest 0.
        aggregation_method (str): How to aggregate the warning days. Defaults to "sum".

    Returns:
        float: Total number of days from first positive prediction to outcome.
    """
    # Generate df with only true positives
    df = pd.DataFrame(
        {
            "id": eval_dataset.ids,
            "pred": eval_dataset.get_predictions_for_positive_rate(
                desired_positive_rate=positive_rate
            )[0],
            "y": eval_dataset.y,
            "pred_timestamps": eval_dataset.pred_timestamps,
            "outcome_timestamps": eval_dataset.outcome_timestamps,
        }
    )

    return get_days_from_first_positive_to_diagnosis_from_df(
        aggregation_method=aggregation_method, df=df
    )


def get_days_from_first_positive_to_diagnosis_from_df(
    aggregation_method: str, df: pd.DataFrame
) -> float:
    df = get_time_from_first_positive_to_diagnosis_df(input_df=df)

    aggregated = df["days_from_pred_to_event"].agg(aggregation_method)
    return aggregated


def get_prop_with_at_least_one_true_positve(
    eval_dataset: EvalDataset, positive_rate: float = 0.5
) -> float:
    """Get proportion of ids with at least one true positive prediction.

    Args:
        eval_dataset (EvalDataset): EvalDataset object.
        positive_rate (float, optional): Takes the top positive_rate% of predicted probabilities and turns them into 1, the rest 0.

    Returns:
        float: Proportion of thresholds with at least one true positive.
    """
    # Generate df with only true positives
    df = get_true_positives(eval_dataset=eval_dataset, positive_rate=positive_rate)

    # Return number of unique ids with at least one true positive
    return df["id"].nunique() / len(set(eval_dataset.ids))


def generate_performance_by_ppr_table(
    eval_dataset: EvalDataset, positive_rates: Sequence[float]
) -> pd.DataFrame:
    """Generates a performance_by_threshold table as either a DataFrame or html
    object.

    Args:
        eval_dataset (EvalDataset): EvalDataset object.
        positive_rates (Sequence[float]): positive_rates to add to the table, e.g. 0.99, 0.98 etc.
            Calculated so that the Xth percentile of predictions are classified as the positive class.

    Returns:
        pd.DataFrame
    """
    rows = []

    # For each percentile, calculate relevant performance metrics
    for positive_rate in positive_rates:
        threshold_metrics = performance_by_ppr(
            eval_dataset=eval_dataset, positive_rate=positive_rate
        )

        threshold_metrics["total_warning_days"] = days_from_first_positive_to_diagnosis(
            eval_dataset=eval_dataset, positive_rate=positive_rate, aggregation_method="sum"
        )

        threshold_metrics["mean_warning_days"] = days_from_first_positive_to_diagnosis(
            eval_dataset=eval_dataset, positive_rate=positive_rate, aggregation_method="mean"
        )

        threshold_metrics["median_warning_days"] = days_from_first_positive_to_diagnosis(
            eval_dataset=eval_dataset, positive_rate=positive_rate, aggregation_method="median"
        )

        threshold_metrics["prop with ≥1 true positive"] = get_prop_with_at_least_one_true_positve(
            eval_dataset=eval_dataset, positive_rate=positive_rate
        )

        threshold_metrics[
            "prop of all events captured"
        ] = get_prop_of_events_captured_from_eval_dataset(
            eval_dataset=eval_dataset, positive_rate=positive_rate
        )

        rows.append(threshold_metrics)

    df = pd.concat(rows)

    df["warning_days_per_false_positive"] = df["total_warning_days"] / df["false_positives"]

    return df
