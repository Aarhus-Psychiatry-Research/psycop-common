"""Get performance by which threshold is used to classify positive. PPR means predicted positive rate, i.e. the proportion of the population that is predicted to be positive."""
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import pandas as pd
import wandb
from psycop.common.model_evaluation.binary.performance_by_ppr.prop_of_all_events_hit_by_true_positive import (
    get_percentage_of_events_captured_from_eval_dataset,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from sklearn.metrics import confusion_matrix


def get_true_positives(
    eval_dataset: EvalDataset,
    positive_rate: float = 0.5,
) -> pd.DataFrame:
    """Get dataframe containing only true positives.

    Args:
        eval_dataset (EvalDataset): EvalDataset object.
        positive_rate (float): Takes the top positive_rate% of predicted probabilities and turns them into 1, the rest 0.

    Returns:
        pd.DataFrame: Dataframe containing only true positives.
    """

    # Generate df
    positives_series, _ = eval_dataset.get_predictions_for_positive_rate(
        desired_positive_rate=positive_rate,
    )

    df = pd.DataFrame(
        {
            "id": eval_dataset.ids,
            "pred": positives_series,
            "y": eval_dataset.y,
            "pred_timestamps": eval_dataset.pred_timestamps,
            "outcome_timestamps": eval_dataset.outcome_timestamps,
        },
    )

    # Keep only true positives
    df["true_positive"] = (df["pred"] == 1) & (df["y"] == 1)

    return df[df["true_positive"]]


def performance_by_ppr(
    eval_dataset: EvalDataset,
    positive_rate: float,
    round_to: int = 2,
) -> pd.DataFrame:
    """Generates a row for a performance_by_threshold table.

    Args:
        eval_dataset (EvalDataset): EvalDataset object.
        positive_rate (float): Takes the top positive_rate% of predicted probabilities and turns them into 1, the rest 0.
        round_to (int): Number of decimal places to round metrics

    Returns:
        pd.DataFrame
    """
    preds, _ = eval_dataset.get_predictions_for_positive_rate(positive_rate)

    conf_matrix = confusion_matrix(eval_dataset.y, preds)

    true_neg = conf_matrix[0][0]
    false_neg = conf_matrix[1][0]
    true_pos = conf_matrix[1][1]
    false_pos = conf_matrix[0][1]

    n_total = true_neg + false_neg + true_pos + false_pos

    true_prevalence = round((true_pos + false_neg) / n_total, round_to)

    positive_rate = round((true_pos + false_pos) / n_total, round_to)
    negative_rate = round((true_neg + false_neg) / n_total, round_to)

    pos_pred_val = round(true_pos / (true_pos + false_pos), round_to)
    neg_pred_val = round(true_neg / (true_neg + false_neg), round_to)

    sens = round(true_pos / (true_pos + false_neg), round_to)
    spec = round(true_neg / (true_neg + false_pos), round_to)

    false_pos_rate = round(false_pos / (true_neg + false_pos), round_to)
    neg_pos_rate = round(false_neg / (true_pos + false_neg), round_to)

    acc = round((true_pos + true_neg) / n_total, round_to)

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
        },
    )

    return metrics_matrix.reset_index(drop=True)


def days_from_first_positive_to_diagnosis(
    eval_dataset: EvalDataset,
    positive_rate: float = 0.5,
    aggregation_method: str = "sum",
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
                desired_positive_rate=positive_rate,
            )[0],
            "y": eval_dataset.y,
            "pred_timestamps": eval_dataset.pred_timestamps,
            "outcome_timestamps": eval_dataset.outcome_timestamps,
        },
    )

    return get_days_from_first_positive_to_diagnosis_from_df(
        aggregation_method=aggregation_method,
        df=df,
    )


def get_days_from_first_positive_to_diagnosis_from_df(
    aggregation_method: str,
    df: pd.DataFrame,
) -> float:
    """Get a dataframe. Easily testable.
    Use the `days_from_first_positive_to_diagnosis` function when you have an eval_dataset.
    """
    # Keep only true positives
    df["true_positive"] = (df["pred"] == 1) & (df["y"] == 1)
    df = df[df["true_positive"]]

    # Find timestamp of first positive prediction
    df["timestamp_first_pos_pred"] = df.groupby("id")["pred_timestamps"].transform(
        "min",
    )
    df = df[df["timestamp_first_pos_pred"].notnull()]

    # Keep only one record per patient
    df = df.drop_duplicates(
        subset=["id", "timestamp_first_pos_pred", "outcome_timestamps"],
    )

    # Calculate warning days
    df["warning_days"] = round(
        (df["outcome_timestamps"] - df["timestamp_first_pos_pred"])
        / np.timedelta64(1, "D"),  # type: ignore
        0,
    )

    df = df[
        [
            "id",
            "timestamp_first_pos_pred",
            "outcome_timestamps",
            "warning_days",
        ]
    ]

    aggregated = df["warning_days"].agg(aggregation_method)
    return aggregated


def get_percent_with_at_least_one_true_positve(
    eval_dataset: EvalDataset,
    positive_rate: float = 0.5,
) -> float:
    """Get proportion of ids with at least one true positive prediction.

    Args:
        eval_dataset (EvalDataset): EvalDataset object.
        positive_rate (float, optional): Takes the top positive_rate% of predicted probabilities and turns them into 1, the rest 0.

    Returns:
        float: Proportion of thresholds with at least one true positive.
    """
    # Generate df with only true positives
    df = get_true_positives(
        eval_dataset=eval_dataset,
        positive_rate=positive_rate,
    )

    # Return number of unique ids with at least one true positive
    return df["id"].nunique() / len(set(eval_dataset.ids)) * 100


def generate_performance_by_ppr_table(
    eval_dataset: EvalDataset,
    positive_rates: Sequence[float],
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
            eval_dataset=eval_dataset,
            positive_rate=positive_rate,
        )

        threshold_metrics["total_warning_days"] = days_from_first_positive_to_diagnosis(
            eval_dataset=eval_dataset,
            positive_rate=positive_rate,
            aggregation_method="sum",
        )

        threshold_metrics["mean_warning_days"] = round(
            days_from_first_positive_to_diagnosis(
                eval_dataset=eval_dataset,
                positive_rate=positive_rate,
                aggregation_method="mean",
            ),
            0,
        )

        threshold_metrics[
            "% with ≥1 true positive"
        ] = get_percent_with_at_least_one_true_positve(
            eval_dataset=eval_dataset,
            positive_rate=positive_rate,
        )

        threshold_metrics[
            "% of all events captured"
        ] = get_percentage_of_events_captured_from_eval_dataset(
            eval_dataset=eval_dataset,
            positive_rate=positive_rate,
        )

        rows.append(threshold_metrics)

    df = pd.concat(rows)

    df["warning_days_per_false_positive"] = (
        df["total_warning_days"] / df["false_positives"]
    ).round(1)

    return df
