from typing import Iterable, Union

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def generate_performance_by_threshold_table(
    labels: Iterable[Union[int, float]],
    pred_probs: Iterable[Union[int, float]],
    threshold_percentiles: Iterable[Union[int, float]],
    ids: Iterable[Union[int, float]],
    pred_timestamps: Iterable[pd.Timestamp],
    outcome_timestamps: Iterable[pd.Timestamp],
) -> pd.DataFrame:
    """Generates a performance_by_threshold table.

    Args:
        labels (Iterable[int, float]): True labels.
        pred_probs (Iterable[int, float]): Predicted probabilities.
        threshold_percentiles (Iterable[float]): Threshold-percentiles to add to the table, e.g. 0.99, 0.98 etc.
            Calculated so that the Xth percentile of predictions are classified as the positive class.
        ids (Iterable[Union[int, float]]): Ids to group on.
        pred_timestamps (Iterable[ pd.Timestamp ]): Timestamp for each prediction time.

    Returns:
        pd.DataFrame
    """

    labels = pd.Series(list(labels))
    pred_probs = pd.Series(list(pred_probs))

    # Calculate percentiles from the pred_probs series
    thresholds = pd.Series(pred_probs).quantile(threshold_percentiles)

    rows = []

    # For each percentile, calculate relevant performance metrics
    for i, threshold_value in enumerate(thresholds):
        threshold_metrics = pd.DataFrame({"threshold": [threshold_percentiles[i]]})

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

    return df.reset_index(drop=True)


def performance_by_threshold(labels, pred_probs, positive_threshold):
    preds = np.where(pred_probs > positive_threshold, 1, 0)

    CM = confusion_matrix(labels, preds)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]

    n = TN + FN + TP + FP

    Prevalence = round((TP + FP) / n, 2)

    PPV = round(TP / (TP + FP), 2)
    NPV = round(TN / (TN + FN), 2)

    Sensitivity = round(TP / (TP + FN), 2)
    Specificity = round(TN / (TN + FP), 2)

    FPR = round(FP / (TN + FP), 2)
    FNR = round(FN / (TP + FN), 2)

    Accuracy = round((TP + TN) / n, 2)

    # Must return lists as values, otherwise pd.Dataframe requires setting indeces
    metrics_matrix = pd.DataFrame(
        {
            "prevalence": [Prevalence],
            "PPV": [PPV],
            "NPV": [NPV],
            "sensitivity": [Sensitivity],
            "specificity": [Specificity],
            "FPR": [FPR],
            "FNR": [FNR],
            "accuracy": [Accuracy],
            "false_positives": [FP],
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
    df = pd.DataFrame(
        {
            "id": ids,
            "pred_probs": pred_probs,
            "pred_timestamps": pred_timestamps,
            "outcome_timestamps": outcome_timestamps,
        },
    )

    timestamp_cols = [col for col in df.columns if "timestamp" in col]

    for col in timestamp_cols:
        df[col] = pd.to_datetime(df[col])

    # Get min date among true positives
    df["true_positive"] = (df["pred_probs"] >= positive_threshold) & (
        df["outcome_timestamps"].notnull()
    )
    df = df[df["true_positive"]]

    df["timestamp_first_pos_pred"] = df.groupby("id")["pred_timestamps"].transform(
        "min",
    )

    df = df.drop_duplicates(
        subset=["id", "timestamp_first_pos_pred", "outcome_timestamps"],
    )

    # Compare to timestamp_t2d_diag
    df["warning_days"] = df["outcome_timestamps"] - df["timestamp_first_pos_pred"]

    df = df[
        [
            "id",
            "timestamp_first_pos_pred",
            "outcome_timestamps",
            "warning_days",
        ]
    ]

    warning_days = (df["warning_days"] / np.timedelta64(1, "D")).astype(int).agg("sum")

    return warning_days
