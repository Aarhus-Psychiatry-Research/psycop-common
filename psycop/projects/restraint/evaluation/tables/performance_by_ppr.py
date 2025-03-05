from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from wasabi import Printer

from psycop.common.model_evaluation.binary.performance_by_ppr.performance_by_ppr import (
    generate_performance_by_ppr_table,
    get_true_positives,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset

from psycop.projects.restraint.evaluation.utils import (
    expand_eval_df_with_extra_cols,
    read_eval_df_from_disk,
)

msg = Printer(timestamp=True)


def _df_to_eval_dataset(df: pd.DataFrame) -> EvalDataset:
    """Convert dataframe to EvalDataset."""
    return EvalDataset(
        ids=df["dw_ek_borger"],
        pred_time_uuids=df["pred_time_uuid"],
        y=df["y"],
        y_hat_probs=df["y_hat_prob"],
        pred_timestamps=df["timestamp"],
        outcome_timestamps=df["timestamp_outcome"],
        age=df["age"],
    )


def _get_num_of_unique_outcome_events(eval_dataset: EvalDataset) -> int:
    df = pd.DataFrame(
        {
            "id": eval_dataset.ids,
            "y": eval_dataset.y,
            "pred_timestamps": eval_dataset.pred_timestamps,
            "outcome_timestamps": eval_dataset.outcome_timestamps,
        }
    )

    num_unique = df[df["y"] == 1][["id", "outcome_timestamps"]].drop_duplicates().shape[0]

    return num_unique


def _get_outcome_for_extended_lookahead(
    eval_dataset: EvalDataset, alternative_lookahead_days: int, postive_rate: float
) -> int:
    """Get outcome labels if time from to outcome is within alternative_lookahead_days form prediction time."""

    y_hat_int, _ = eval_dataset.get_predictions_for_positive_rate(
        desired_positive_rate=postive_rate
    )

    df = pd.DataFrame(
        {
            "id": eval_dataset.ids,
            "y": eval_dataset.y,
            "pred_timestamps": eval_dataset.pred_timestamps,
            "outcome_timestamps": eval_dataset.outcome_timestamps,
            "pred": y_hat_int,
        }
    )

    df["days_from_pred_to_outcome"] = (df["outcome_timestamps"] - df["pred_timestamps"]).dt.days

    true_positive_given_alternative_lookahead = df[
        (df["pred"] == 1) & (df["days_from_pred_to_outcome"] <= alternative_lookahead_days)
    ]

    return len(true_positive_given_alternative_lookahead)


def _get_admission_level_model_behavior_with_extended_lookahead(
    eval_dataset: EvalDataset, alternative_lookahead_days: int, postive_rate: float
) -> int:
    """If a positive prediction is issued within the alternative_lookahead_days, the admission level is considered to be predicted as true positive.
    If an admission with an outcome does not have a single positive prediction within the alternative_lookahead_days, the admission level is considered to be predicted as false negative.
    If an admission with an outcome does not have a single positive prediction within the entire admission, the admission level is considered to be predicted as a very false negative.
    If an admission without an outcome has a positive prediction, the admission level is considered to be predicted as false positive."""

    y_hat_int, _ = eval_dataset.get_predictions_for_positive_rate(
        desired_positive_rate=postive_rate
    )

    df = pd.DataFrame(
        {
            "id": eval_dataset.ids,
            "y": eval_dataset.y,
            "pred_timestamps": eval_dataset.pred_timestamps,
            "outcome_timestamps": eval_dataset.outcome_timestamps,
            "pred": y_hat_int,
        }
    )

    df["days_from_pred_to_outcome"] = (df["outcome_timestamps"] - df["pred_timestamps"]).dt.days

    # Get admissions with outcomes
    admissions_with_outcomes = df[df["y"] == 1][["id", "outcome_timestamps"]].drop_duplicates()

    # Get admissions without outcomes
    admissions_without_outcomes = df[df["y"] == 0][["id"]].drop_duplicates()

    # Get admissions with positive predictions
    admissions_with_positive_predictions = df[df["pred"] == 1][["id"]].drop_duplicates()

    # Get admissions with positive predictions within the alternative_lookahead_days
    admissions_with_positive_predictions_within_lookahead = df[
        (df["pred"] == 1) & (df["days_from_pred_to_outcome"] <= alternative_lookahead_days)
    ][["id"]].drop_duplicates()

    # Get admissions with positive predictions within the entire admission
    admissions_with_positive_predictions_within_admission = df[
        (df["pred"] == 1) & (df["days_from_pred_to_outcome"] >= 0)
    ][["id"]].drop_duplicates()

    # Get admissions with no positive predictions
    admissions_with_no_positive_predictions = df[df["pred"] == 0][["id"]].drop_duplicates()

    # Get admissions with no positive predictions within the entire admission
    admissions_with_no_positive_predictions_within_admission = df[
        (df["pred"] == 0) & (df["days_from_pred_to_outcome"] >= 0)
    ][["id"]].drop_duplicates()

    # Get admissions with no positive predictions within the alternative_lookahead_days
    admissions_with_no_positive_predictions_within_lookahead = df[
        (df["pred"] == 0) & (df["days_from_pred_to_outcome"] <= alternative_lookahead_days)
    ][["id"]].drop_duplicates()


def _get_number_of_outcome_events_with_at_least_one_true_positve(
    eval_dataset: EvalDataset, positive_rate: float, min_alert_days: None | int = 30
) -> float:
    """Get number of outcomes with a prediction time that has at least one true positive prediction.

    Args:
        eval_dataset (EvalDataset): EvalDataset object.
        positive_rate (float, optional): Takes the top positive_rate% of predicted probabilities and turns them into 1, the rest 0.
        min_alert_days (None | int, optional): Minimum number of days a positive prediction must be made before the outcome event for it to be considered actionable/beneficial. Defaults to 30.

    Returns:
        float: Number of outcomes with at least one true positive.
    """

    # Generate df with only true positives
    tp_df = get_true_positives(eval_dataset, positive_rate)

    # If min alert days is not None, creatre a column that calculates the number of days from pred_timestamps to outcome_timestamps
    if min_alert_days is not None:
        tp_df["days_from_pred_to_outcome"] = (
            tp_df["outcome_timestamps"] - tp_df["pred_timestamps"]
        ).dt.days

        # group by id and outcome_timestamps and sort by days_from_pred_to_outcome from highest to lowest
        tp_df = tp_df.sort_values(
            by=["id", "outcome_timestamps", "days_from_pred_to_outcome"],
            ascending=[True, True, False],
        )

        # if the first row of days_from_pred_to_outcome for each id and outcome id is smaller than min_alert_days, drop all rows for that id and outcome id
        tp_df = tp_df.groupby(["id", "outcome_timestamps"]).filter(
            lambda x: x["days_from_pred_to_outcome"].iloc[0] >= min_alert_days
        )

    # Return number of outcome events with at least one true positives
    return tp_df[["id", "outcome_timestamps"]].drop_duplicates().shape[0]


def format_with_thousand_separator(num: int) -> str:
    return f"{num:,.0f}"


def format_prop_as_percent(num: float) -> str:
    output = f"{num:.1%}"

    return output


def clean_up_performance_by_ppr(table: pd.DataFrame) -> pd.DataFrame:
    df = table

    output_df = df.drop(
        [
            "total_warning_days",
            "warning_days_per_false_positive",
            "negative_rate",
            "mean_warning_days",
            "median_warning_days",
            "prop with ≥1 true positive",
            "prop of all events captured",
        ],
        axis=1,
    )

    renamed_df = output_df.rename(
        {
            "positive_rate": "Predicted positive rate",
            "true_prevalence": "True prevalence",
            "sensitivity": "Sens",
            "specificity": "Spec",
            "accuracy": "Acc",
            "true_positives": "TP",
            "true_negatives": "TN",
            "false_positives": "FP",
            "false_negatives": "FN",
            "f1": "F1",
            "mcc": "MCC",
        },
        axis=1,
    )

    # Handle proportion columns
    prop_cols = [c for c in renamed_df.columns if renamed_df[c].dtype == "float64"]
    for c in prop_cols:
        renamed_df[c] = renamed_df[c].apply(format_prop_as_percent)

    # Handle count columns
    count_cols = [c for c in renamed_df.columns if renamed_df[c].dtype == "int64"]
    for col in count_cols:
        renamed_df[col] = renamed_df[col].apply(format_with_thousand_separator)

    renamed_df["Median days from first positive to outcome"] = round(df["median_warning_days"], 1)

    return renamed_df


def restraint_output_performance_by_ppr(
    eval_df: pd.DataFrame,
    eval_dir: str,
    save: bool = True,
    positive_rates: Sequence[float] = [0.5, 0.2, 0.1, 0.075, 0.05, 0.04, 0.03, 0.02, 0.01],
    min_alert_days: None | int = None,
    alternative_lookahead_days: int = 10,
) -> pd.DataFrame | None:
    eval_dataset = _df_to_eval_dataset(eval_df)

    df: pd.DataFrame = generate_performance_by_ppr_table(  # type: ignore
        eval_dataset=eval_dataset, positive_rates=positive_rates
    )

    df["Total number of unique outcome events"] = _get_num_of_unique_outcome_events(
        eval_dataset=eval_dataset
    )

    df["Number of positive outcomes in test set (TP+FN)"] = (
        df["true_positives"] + df["false_negatives"]
    )

    df["Number of unique outcome events detected ≥1"] = [
        _get_number_of_outcome_events_with_at_least_one_true_positve(
            eval_dataset=eval_dataset, positive_rate=pos_rate, min_alert_days=min_alert_days
        )
        for pos_rate in positive_rates
    ]

    df["Prop. of unique outcome events detected ≥1"] = round(
        df["Number of unique outcome events detected ≥1"]
        / df["Total number of unique outcome events"],
        3,
    )

    df["Number of true positives if lookahead is 30 days"] = [
        _get_outcome_for_extended_lookahead(eval_dataset, alternative_lookahead_days, pos_rate)
        for pos_rate in positive_rates
    ]

    df = clean_up_performance_by_ppr(df)

    if save:
        df.to_excel(Path(eval_dir), index=False)
        return None

    return df


if __name__ == "__main__":
    eval_dir = (
        "E:/shared_resources/restraint/eval_runs/restraint_split_tuning_best_run_evaluated_on_test"
    )

    restraint_output_performance_by_ppr(
        expand_eval_df_with_extra_cols(read_eval_df_from_disk(eval_dir)), eval_dir
    )
