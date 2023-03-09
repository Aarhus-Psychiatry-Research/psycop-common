"""Get performance by which threshold is used to classify positive."""
from collections.abc import Iterable
from typing import Optional, Union

import numpy as np
import pandas as pd
import wandb
from psycop_model_training.model_eval.dataclasses import EvalDataset
from sklearn.metrics import confusion_matrix


def get_true_positives(
    eval_dataset: EvalDataset,
    positive_rate_threshold: Optional[float] = 0.5,
):
    """Get dataframe containing only true positives.

    Args:
        eval_dataset (EvalDataset): EvalDataset object.
        positive_rate_threshold (float, optional): Threshold above which patients are classified as positive. Defaults to 0.5.

    Returns:
        pd.DataFrame: Dataframe containing only true positives.
    """

    # Generate df
    df = pd.DataFrame(
        {
            "id": eval_dataset.ids,
            "pred_probs": eval_dataset.y_hat_probs,
            "pred_timestamps": eval_dataset.pred_timestamps,
            "outcome_timestamps": eval_dataset.outcome_timestamps,
        },
    )

    # Keep only true positives
    df["true_positive"] = (df["pred_probs"] >= positive_rate_threshold) & (
        df["outcome_timestamps"].notnull()
    )

    return df[df["true_positive"]]


def performance_by_threshold(  # pylint: disable=too-many-locals
    eval_dataset: EvalDataset,
    positive_threshold: float,
    round_to: int = 4,
) -> pd.DataFrame:
    """Generates a row for a performance_by_threshold table.

    Args:
        eval_dataset (EvalDataset): EvalDataset object.
        positive_threshold (float): Threshold for a probability to be
            labelled as "positive".
        round_to (int): Number of decimal places to round metrics

    Returns:
        pd.DataFrame
    """
    preds = np.where(eval_dataset.y_hat_probs > positive_threshold, 1, 0)  # type: ignore

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
    positive_rate_threshold: Optional[float] = 0.5,
    aggregation_method: Optional[str] = "sum",
) -> float:
    """Calculate number of days from the first positive prediction to the
    patient's outcome timestamp.

    Args:
        eval_dataset (EvalDataset): EvalDataset object.
        positive_rate_threshold (float, optional): Threshold above which patients are classified as positive. Defaults to 0.5.
        aggregation_method (str, optional): How to aggregate the warning days. Defaults to "sum".

    Returns:
        float: Total number of days from first positive prediction to outcome.
    """
    # Generate df with only true positives
    df = get_true_positives(
        eval_dataset=eval_dataset,
        positive_rate_threshold=positive_rate_threshold,
    )

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

    return df["warning_days"].agg(aggregation_method)


def prop_with_at_least_one_true_positve(
    eval_dataset: EvalDataset,
    positive_rate_threshold: Optional[float] = 0.5,
) -> float:
    """Get proportion of patients with at least one true positive prediction.

    Args:
        eval_dataset (EvalDataset): EvalDataset object.
        positive_rate_threshold (float, optional): Threshold above which patients are classified as positive. Defaults to 0.5.

    Returns:
        float: Proportion of thresholds with at least one true positive.
    """
    # Generate df with only true positives
    df = get_true_positives(
        eval_dataset=eval_dataset,
        positive_rate_threshold=positive_rate_threshold,
    )

    # Return number of unique patients with at least one true positive
    return round(df["id"].nunique() / len(set(eval_dataset.ids)), 4)


def generate_performance_by_positive_rate_table(
    eval_dataset: EvalDataset,
    positive_rate_thresholds: Iterable[Union[int, float]],
    pred_proba_thresholds: Iterable[float],
    output_format: Optional[str] = "df",
) -> Union[pd.DataFrame, str, wandb.Table]:
    """Generates a performance_by_threshold table as either a DataFrame or html
    object.

    Args:
        eval_dataset (EvalDataset): EvalDataset object.
        positive_rate_thresholds (Sequence[float]): Positive_rate_thresholds to add to the table, e.g. 0.99, 0.98 etc.
            Calculated so that the Xth percentile of predictions are classified as the positive class.
        pred_proba_thresholds (Sequence[float]): Thresholds above which predictions are classified as positive.
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
            eval_dataset=eval_dataset,
            positive_threshold=threshold_value,
        )

        threshold_metrics[  # pylint: disable=unsupported-assignment-operation
            "total_warning_days"
        ] = days_from_first_positive_to_diagnosis(
            eval_dataset=eval_dataset,
            positive_rate_threshold=threshold_value,
            aggregation_method="sum",
        )

        threshold_metrics[  # pylint: disable=unsupported-assignment-operation
            "mean_warning_days"
        ] = round(
            days_from_first_positive_to_diagnosis(
                eval_dataset=eval_dataset,
                positive_rate_threshold=threshold_value,
                aggregation_method="mean",
            ),
            0,
        )

        threshold_metrics[  # pylint: disable=unsupported-assignment-operation
            "prop_with_at_least_one_true_positive"
        ] = prop_with_at_least_one_true_positve(
            eval_dataset=eval_dataset,
            positive_rate_threshold=threshold_value,
        )

        rows.append(threshold_metrics)

    df = pd.concat(rows)

    df["warning_days_per_false_positive"] = (
        df["total_warning_days"] / df["false_positives"]
    ).round(1)

    if output_format == "html":
        return df.reset_index(drop=True).to_html()
    if output_format == "df":
        return df.reset_index(drop=True)
    if output_format == "wandb_table":
        return wandb.Table(dataframe=df.reset_index(drop=True))

    raise ValueError("Output format does not match anything that is allowed")
