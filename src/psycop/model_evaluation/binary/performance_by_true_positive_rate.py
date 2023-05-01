"""Get performance by which threshold is used to classify positive."""
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import pandas as pd
import wandb
from psycop_model_training.training_output.dataclasses import EvalDataset
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
            "pred_timestamps": eval_dataset.pred_timestamps,
            "outcome_timestamps": eval_dataset.outcome_timestamps,
        },
    )

    # Keep only true positives
    df["true_positive"] = (df["pred"] == 1) & (df["outcome_timestamps"].notnull())

    return df[df["true_positive"]]


def performance_by_positive_rate(
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
    df = get_true_positives(
        eval_dataset=eval_dataset,
        positive_rate=positive_rate,
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

    return df["warning_days"].agg(aggregation_method)


def prop_with_at_least_one_true_positve(
    eval_dataset: EvalDataset,
    positive_rate: float = 0.5,
) -> float:
    """Get proportion of patients with at least one true positive prediction.

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

    # Return number of unique patients with at least one true positive
    return round(df["id"].nunique() / len(set(eval_dataset.ids)), 4)


def generate_performance_by_positive_rate_table(
    eval_dataset: EvalDataset,
    positive_rates: Sequence[float],
    output_format: Optional[str] = "df",
) -> Union[pd.DataFrame, str, wandb.Table]:
    """Generates a performance_by_threshold table as either a DataFrame or html
    object.

    Args:
        eval_dataset (EvalDataset): EvalDataset object.
        positive_rates (Sequence[float]): positive_rates to add to the table, e.g. 0.99, 0.98 etc.
            Calculated so that the Xth percentile of predictions are classified as the positive class.
        output_format (str, optional): Format to output - either "df" or "wandb_table". Defaults to "df".

    Returns:
        pd.DataFrame
    """
    rows = []

    # For each percentile, calculate relevant performance metrics
    for positive_rate in positive_rates:
        threshold_metrics = performance_by_positive_rate(
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
            "prop_with_at_least_one_true_positive"
        ] = prop_with_at_least_one_true_positve(
            eval_dataset=eval_dataset,
            positive_rate=positive_rate,
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
