import logging
from collections.abc import Sequence
from functools import partial
from multiprocessing import Pool
from typing import NewType

import pandas as pd
import polars as pl

from psycop.common.global_utils.cache import shared_cache
from psycop.common.global_utils.mlflow.mlflow_data_extraction import EvalFrame, MlflowClientWrapper
from psycop.common.model_evaluation.confusion_matrix.confusion_matrix import (
    get_confusion_matrix_cells_from_df,
)
from psycop.common.model_training.training_output.dataclasses import (
    get_predictions_for_positive_rate,
)
from psycop.projects.ect.feature_generation.cohort_definition.outcome_specification.combined import (
    add_first_ect_time_after_prediction_time,
)
from psycop.projects.ect.model_evaluation.performance_by_ppr.days_from_first_positive_to_event import (
    days_from_first_positive_to_event,
)
from psycop.projects.ect.model_evaluation.uuid_parsers import (
    parse_dw_ek_borger_from_uuid,
    parse_timestamp_from_uuid,
)


def _performance_by_ppr_row(eval_df: pl.DataFrame, positive_rate: float) -> pl.DataFrame:
    logging.info(f"Generating performance by PPR row with positive rate {positive_rate}")
    pd_df = eval_df.to_pandas()
    preds, _ = get_predictions_for_positive_rate(
        desired_positive_rate=positive_rate, y_hat_probs=pd_df["y_hat_prob"]
    )

    conf_matrix_df = pd.DataFrame({"pred": preds, "true": pd_df["y"]})

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
        }
    )

    return pl.from_pandas(metrics_matrix.reset_index(drop=True))


def _get_prop_with_at_least_one_true_positve(
    eval_df: pl.DataFrame, positive_rate: float = 0.5
) -> float:
    logging.info("Getting proportion with at least one true positive")
    base_df = parse_timestamp_from_uuid(parse_dw_ek_borger_from_uuid(eval_df)).to_pandas()

    df = pd.DataFrame(
        {
            "id": base_df["dw_ek_borger"],
            "pred": get_predictions_for_positive_rate(
                desired_positive_rate=positive_rate, y_hat_probs=eval_df["y_hat_prob"].to_pandas()
            )[0],
            "y": base_df["y"],
            "pred_timestamps": base_df["timestamp"],
            "outcome_timestamps": base_df["timestamp_outcome"],
        }
    )

    # Keep only true positives
    df["true_positive"] = (df["pred"] == 1) & (df["y"] == 1)
    true_positives = df[df["true_positive"]]

    return true_positives["id"].nunique() / df["id"].nunique()


def get_percentage_of_events_captured(eval_df: pl.DataFrame, positive_rate: float) -> float:
    logging.info("Calculating proportion of events hit by a positive prediction")
    # Get all patients with at least one event and at least one positive prediction
    base_df = parse_dw_ek_borger_from_uuid(eval_df).to_pandas()
    base_df["pred"] = get_predictions_for_positive_rate(
        desired_positive_rate=positive_rate, y_hat_probs=base_df["y_hat_prob"]
    )[0]

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


def _calculate_row(positive_rate: float, eval_df: pl.DataFrame) -> pl.DataFrame:
    threshold_metrics = _performance_by_ppr_row(
        eval_df=eval_df, positive_rate=positive_rate
    ).with_columns(
        pl.lit(
            _get_prop_with_at_least_one_true_positve(eval_df=eval_df, positive_rate=positive_rate)
        ).alias("prop with ≥ 1 true positive"),
        pl.lit(
            get_percentage_of_events_captured(eval_df=eval_df, positive_rate=positive_rate)
        ).alias("prop_of_all_events_captured"),
        pl.lit(
            days_from_first_positive_to_event(
                eval_dataset=eval_df, positive_rate=positive_rate, aggregation_method="sum"
            )
        ).alias("total_warning_days"),
        pl.lit(
            days_from_first_positive_to_event(
                eval_dataset=eval_df, positive_rate=positive_rate, aggregation_method="mean"
            )
        ).alias("mean_warning_days"),
        pl.lit(
            days_from_first_positive_to_event(
                eval_dataset=eval_df, positive_rate=positive_rate, aggregation_method="median"
            )
        ).alias("median_warning_days"),
    )

    return threshold_metrics


PerformanceByPPRModel = NewType("PerformanceByPPRModel", pl.DataFrame)


@shared_cache().cache()
def performance_by_ppr_model(
    eval_df: EvalFrame, positive_rates: Sequence[float]
) -> PerformanceByPPRModel:
    eval_df_with_correct_outcome_timestamp = add_first_ect_time_after_prediction_time(eval_df.frame)

    rows = Pool(len(positive_rates)).map(
        partial(_calculate_row, eval_df=eval_df_with_correct_outcome_timestamp), positive_rates
    )

    df = pl.concat(rows).with_columns(
        (pl.col("total_warning_days") / pl.col("false_positives")).alias(
            "warning_days_per_false_positive"
        )
    )
    return PerformanceByPPRModel(df)


if __name__ == "__main__":
    MAIN_METRIC = "all_oof_BinaryAUROC"

    eval_df = (
        MlflowClientWrapper()
        .get_best_run_from_experiment(
            experiment_name="ECT hparam, structured_text, xgboost, no lookbehind filter",
            metric=MAIN_METRIC,
        )
        .eval_frame()
    )

    pers = performance_by_ppr_model(eval_df, [0.01, 0.02, 0.03, 0.04, 0.05])
    # add correct outcome timestamp
