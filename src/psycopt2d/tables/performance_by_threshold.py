from typing import Iterable, Union

import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import confusion_matrix


def generate_performance_by_threshold_table(
    labels: Iterable[Union[int, float]],
    pred_probs: Iterable[Union[int, float]],
    threshold_percentiles: Iterable[Union[int, float]],
    pred_proba_thresholds: Iterable[float],
    ids: Iterable[Union[int, float]],
    pred_timestamps: Iterable[pd.Timestamp],
    outcome_timestamps: Iterable[pd.Timestamp],
    output_format: str = "wandb_table",
) -> Union[pd.DataFrame, str]:
    """Generates a performance_by_threshold table as either a DataFrame or html
    object.

    Args:
        labels (Iterable[int, float]): True labels.
        pred_probs (Iterable[int, float]): Predicted probabilities.
        threshold_percentiles (Iterable[float]): Threshold-percentiles to add to the table, e.g. 0.99, 0.98 etc.
            Calculated so that the Xth percentile of predictions are classified as the positive class.
        pred_proba_thresholds (Iterable[float]): Thresholds above which predictions are classified as positive.
        ids (Iterable[Union[int, float]]): Ids to group on.
        pred_timestamps (Iterable[ pd.Timestamp ]): Timestamp for each prediction time.
        output_format (str, optional): Format to output - either "df" or "html". Defaults to "df".

    Returns:
        pd.DataFrame
    """

    rows = []

    # For each percentile, calculate relevant performance metrics
    for i, threshold_value in enumerate(pred_proba_thresholds):
        threshold_metrics = pd.DataFrame(
            {"threshold_percentile": [threshold_percentiles[i]]},
        )

        threshold_metrics = pd.concat(
            [
                threshold_metrics,
                performance_by_threshold(
                    labels=labels,
                    pred_probs=pred_probs,
                    positive_threshold=threshold_value,
                ),
            ],
            axis=1,
        )

        threshold_metrics["warning_days"] = days_from_positive_to_diagnosis(
            ids=ids,
            pred_probs=pred_probs,
            pred_timestamps=pred_timestamps,
            outcome_timestamps=outcome_timestamps,
            positive_threshold=threshold_value,
        )

        rows.append(threshold_metrics)

    df = pd.concat(rows)
    df["warning_days_per_false_positive"] = round(
        (threshold_metrics["warning_days"] / threshold_metrics["false_positives"]),
        2,
    )

    if output_format == "html":
        return df.reset_index(drop=True).to_html()
    elif output_format == "df":
        return df.reset_index(drop=True)
    elif output_format == "wandb_table":
        return wandb.Table(dataframe=df.reset_index(drop=True))
    else:
        raise ValueError("Output format does not match anything that is allowed")


def performance_by_threshold(
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

    CM = confusion_matrix(labels, preds)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]

    n_total = TN + FN + TP + FP

    true_prevalence = round((TP + FN) / n_total, round_to)

    positive_rate = round((TP + FP) / n_total, round_to)
    negative_rate = round((TN + FN) / n_total, round_to)

    PPV = round(TP / (TP + FP), round_to)
    NPV = round(TN / (TN + FN), round_to)

    Sensitivity = round(TP / (TP + FN), round_to)
    Specificity = round(TN / (TN + FP), round_to)

    FPR = round(FP / (TN + FP), round_to)
    FNR = round(FN / (TP + FN), round_to)

    Accuracy = round((TP + TN) / n_total, round_to)

    # Must return lists as values, otherwise pd.Dataframe requires setting indeces
    metrics_matrix = pd.DataFrame(
        {
            "true_prevalence": [true_prevalence],
            "positive_rate": [positive_rate],
            "negative_rate": [negative_rate],
            "PPV": [PPV],
            "NPV": [NPV],
            "sensitivity": [Sensitivity],
            "specificity": [Specificity],
            "FPR": [FPR],
            "FNR": [FNR],
            "accuracy": [Accuracy],
            "true_positives": [TP],
            "true_negatives": [TN],
            "false_positives": [FP],
            "false_negatives": [FN],
        },
    )

    return metrics_matrix.reset_index(drop=True)


def days_from_positive_to_diagnosis(
    ids: Iterable[Union[float, str]],
    pred_probs: Iterable[Union[float, str]],
    pred_timestamps: Iterable[pd.Timestamp],
    outcome_timestamps: Iterable[pd.Timestamp],
    positive_threshold: float = 0.5,
) -> float:
    """Calculate number of days from the first positive prediction to the
    patient's outcome timestamp.

    Args:
        ids (Iterable[Union[float, str]]): _description_
        pred_probs (Iterable[Union[float, str]]): _description_
        pred_timestamps (Iterable[pd.Timestamp]): _description_
        outcome_timestamps (Iterable[pd.Timestamp]): _description_
        positive_threshold (float, optional): _description_. Defaults to 0.5.

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
    df["true_positive"] = (df["pred_probs"] >= positive_threshold) & (
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

    warning_days = df["warning_days"].agg("sum")

    return warning_days
