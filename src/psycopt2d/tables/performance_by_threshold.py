"""Get performance by which threshold is used to classify positive."""
from collections.abc import Iterable
from typing import Optional, Union

import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import confusion_matrix


def generate_performance_by_positive_rate_table(
    labels: Iterable[int],
    pred_probs: Iterable[float],
    positive_rate_thresholds: Iterable[Union[int, float]],
    pred_proba_thresholds: Iterable[float],
    ids: Iterable[Union[int, float]],
    pred_timestamps: Iterable[pd.Timestamp],
    outcome_timestamps: Iterable[pd.Timestamp],
    output_format: Optional[str] = "wandb_table",
) -> Union[pd.DataFrame, str]:
    """Generates a performance_by_threshold table as either a DataFrame or html
    object.

    Args:
        labels (Iterable[int]): True labels.
        pred_probs (Iterable[float]): Predicted probabilities.
        positive_rate_thresholds (Iterable[float]): Positive_rate_thresholds to add to the table, e.g. 0.99, 0.98 etc.
            Calculated so that the Xth percentile of predictions are classified as the positive class.
        pred_proba_thresholds (Iterable[float]): Thresholds above which predictions are classified as positive.
        ids (Iterable[Union[int, float]]): Ids to group on.
        pred_timestamps (Iterable[ pd.Timestamp ]): Timestamp for each prediction time.
        outcome_timestamps (Iterable[pd.Timestamp]): Timestamp for each outcome time.
        output_format (str, optional): Format to output - either "df" or "wandb_table". Defaults to "df".

    Returns:
        pd.DataFrame
    """

    # Round decimals to percent, e.g. 0.99 -> 99%
    if min(positive_rate_thresholds) < 1:
        positive_rate_thresholds = [x * 100 for x in positive_rate_thresholds]

    rows = []

    # For each percentile, calculate relevant performance metrics
    for threshold_value in pred_proba_thresholds:
        threshold_metrics = performance_by_threshold(
            labels=labels,
            pred_probs=pred_probs,
            positive_threshold=threshold_value,
        )

        threshold_metrics[  # pylint: disable=unsupported-assignment-operation
            "total_warning_days"
        ] = days_from_first_positive_to_diagnosis(
            ids=ids,
            pred_probs=pred_probs,
            pred_timestamps=pred_timestamps,
            outcome_timestamps=outcome_timestamps,
            positive_rate_threshold=threshold_value,
            aggregation_method="sum",
        )

        threshold_metrics[  # pylint: disable=unsupported-assignment-operation
            "mean_warning_days"
        ] = round(
            days_from_first_positive_to_diagnosis(
                ids=ids,
                pred_probs=pred_probs,
                pred_timestamps=pred_timestamps,
                outcome_timestamps=outcome_timestamps,
                positive_rate_threshold=threshold_value,
                aggregation_method="mean",
            ),
            0,
        )

        rows.append(threshold_metrics)

    df = pd.concat(rows)

    df["warning_days_per_false_positive"] = (
        df["total_warning_days"] / df["false_positives"]
    ).round(1)

    if output_format == "html":
        return df.reset_index(drop=True).to_html()
    elif output_format == "df":
        return df.reset_index(drop=True)
    elif output_format == "wandb_table":
        return wandb.Table(dataframe=df.reset_index(drop=True))
    else:
        raise ValueError("Output format does not match anything that is allowed")


def performance_by_threshold(  # pylint: disable=too-many-locals
    labels: Iterable[int],
    pred_probs: Iterable[float],
    positive_threshold: float,
    round_to: int = 4,
) -> pd.DataFrame:
    """Generates a row for a performance_by_threshold table.

    Args:
        labels (Iterable[int]): True labels.
        pred_probs (Iterable[float]): Model prediction probabilities.
        positive_threshold (float): Threshold for a probability to be
            labelled as "positive".
        round_to (int): Number of decimal places to round metrics

    Returns:
        pd.DataFrame
    """
    preds = np.where(pred_probs > positive_threshold, 1, 0)

    conf_matrix = confusion_matrix(labels, preds)

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

    # Must return lists as values, otherwise pd.Dataframe requires setting indeces
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
    ids: Iterable[Union[float, str]],
    pred_probs: Iterable[Union[float, str]],
    pred_timestamps: Iterable[pd.Timestamp],
    outcome_timestamps: Iterable[pd.Timestamp],
    positive_rate_threshold: Optional[float] = 0.5,
    aggregation_method: Optional[str] = "sum",
) -> float:
    """Calculate number of days from the first positive prediction to the
    patient's outcome timestamp.

    Args:
        ids (Iterable[Union[float, str]]): Iterable of patient IDs.
        pred_probs (Iterable[Union[float, str]]): Predicted probabilities.
        pred_timestamps (Iterable[pd.Timestamp]): Timestamps for each prediction time.
        outcome_timestamps (Iterable[pd.Timestamp]): Timestamps of patient outcome.
        positive_rate_threshold (float, optional): Threshold above which patients are classified as positive. Defaults to 0.5.
        aggregation_method (str, optional): How to aggregate the warning days. Defaults to "sum".

    Returns:
        float: _description_
    """
    # Generate df
    df = pd.DataFrame(
        {
            "id": ids,
            "pred_probs": pred_probs,
            "pred_timestamps": pred_timestamps,
            "outcome_timestamps": outcome_timestamps,
        },
    )

    # Keep only true positives
    df["true_positive"] = (df["pred_probs"] >= positive_rate_threshold) & (
        df["outcome_timestamps"].notnull()
    )
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
        / np.timedelta64(1, "D"),
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

    warning_days = df["warning_days"].agg(aggregation_method)

    return warning_days
