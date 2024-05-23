from collections.abc import Sequence
from typing import NewType

import numpy as np
import pandas as pd
import polars as pl

from psycop.common.cohort_definition import OutcomeTimestampFrame, PredictionTimeFrame
from psycop.common.global_utils.mlflow.mlflow_data_extraction import EvalDF
from psycop.common.model_evaluation.confusion_matrix.confusion_matrix import (
    get_confusion_matrix_cells_from_df,
)
from psycop.common.model_training.training_output.dataclasses import (
    get_predictions_for_positive_rate,
)
from psycop.common.types.validated_frame import ValidatedFrame
from psycop.projects.cvd.model_evaluation.single_run.performance_by_ppr.days_from_first_positive_to_event import (
    days_from_first_positive_to_event,
)
from psycop.projects.cvd.model_evaluation.uuid_parsers import (
    parse_dw_ek_borger_from_uuid,
    parse_timestamp_from_uuid,
)


def _performance_by_ppr_row(eval_df: EvalDF, positive_rate: float) -> pl.DataFrame:
    pd_df = eval_df.frame.to_pandas()
    preds, _ = get_predictions_for_positive_rate(
        desired_positive_rate=positive_rate, y_hat_probs=pd_df[eval_df.y_hat_prob_col_name]
    )

    conf_matrix_df = pd.DataFrame({"pred": preds, "true": pd_df[eval_df.y_col_name]})

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

    return pl.from_pandas(metrics_matrix.reset_index(drop=True))


def _get_prop_with_at_least_one_true_positve(
    eval_df: EvalDF,
    outcome_timestamps: OutcomeTimestampFrame,
    prediction_timestamps: PredictionTimeFrame,
    positive_rate: float = 0.5,
) -> float:
    base_df = parse_timestamp_from_uuid(parse_dw_ek_borger_from_uuid(eval_df.frame)).to_pandas()

    df = pd.DataFrame(
        {
            "id": base_df["dw_ek_borger"],
            "pred": get_predictions_for_positive_rate(
                desired_positive_rate=positive_rate,
                y_hat_probs=eval_df.frame[eval_df.y_hat_prob_col_name].to_pandas(),
            ),
            "y": base_df["y"],
            "pred_timestamps": prediction_timestamps.frame[
                prediction_timestamps.timestamp_col_name
            ].to_pandas(),
            "outcome_timestamps": outcome_timestamps.frame[
                outcome_timestamps.timestamp_col_name
            ].to_pandas(),
        }
    )

    # Keep only true positives
    df["true_positive"] = (df["pred"] == 1) & (df["y"] == 1)
    true_positives = df[df["true_positive"]]

    return true_positives["id"].nunique() / len(set(true_positives))


def get_percentage_of_events_captured(df: EvalDF, positive_rate: float) -> float:
    # Get all patients with at least one event and at least one positive prediction
    base_df = parse_dw_ek_borger_from_uuid(df.frame).to_pandas()
    base_df["pred"] = get_predictions_for_positive_rate(
        desired_positive_rate=positive_rate, y_hat_probs=base_df["y_hat_prob"]
    )

    df_patients_with_events = (
        base_df.groupby("dw_ek_borger")
        .filter(lambda x: x["y"].sum() > 0)
        .groupby("dw_ek_borger")
        .head(1)  # type: ignore
    )

    df_events_captured = df_patients_with_events.groupby("dw_ek_borger").filter(
        lambda x: x["pred"].sum() > 0
    )

    return len(df_events_captured) / len(df_patients_with_events)


PerformanceByPPRModel = NewType("PerformanceByPPRModel", pl.DataFrame)


def performance_by_ppr_model(
    eval_df: EvalDF,
    positive_rates: Sequence[float],
    pred_timestamps: PredictionTimeFrame,
    outcome_timestamps: OutcomeTimestampFrame,
) -> PerformanceByPPRModel:
    rows = []

    for positive_rate in positive_rates:
        threshold_metrics = _performance_by_ppr_row(eval_df=eval_df, positive_rate=positive_rate)
        threshold_metrics["total_warning_days"] = days_from_first_positive_to_event(
            eval_dataset=eval_df,
            outcome_timestamps=outcome_timestamps,
            positive_rate=positive_rate,
            aggregation_method="sum",
        )
        threshold_metrics["mean_warning_days"] = days_from_first_positive_to_event(
            eval_dataset=eval_df,
            outcome_timestamps=outcome_timestamps,
            positive_rate=positive_rate,
            aggregation_method="mean",
        )
        threshold_metrics["median_warning_days"] = days_from_first_positive_to_event(
            eval_dataset=eval_df,
            outcome_timestamps=outcome_timestamps,
            positive_rate=positive_rate,
            aggregation_method="median",
        )
        threshold_metrics["pop with ≥ 1 true positive"] = _get_prop_with_at_least_one_true_positve(
            eval_df=eval_df,
            outcome_timestamps=outcome_timestamps,
            prediction_timestamps=pred_timestamps,
            positive_rate=positive_rate,
        )
        threshold_metrics["prop_of_all_events_captured"] = _get_prop_with_at_least_one_true_positve(
            eval_df=eval_df,
            outcome_timestamps=outcome_timestamps,
            prediction_timestamps=pred_timestamps,
            positive_rate=positive_rate,
        )
        rows.append(threshold_metrics)

    df = pd.concat(rows)
    df["warning_days_per_false_positive"] = df["total_warning_days"] / df["false_positives"]
    return PerformanceByPPRModel(pl.from_pandas(df))
