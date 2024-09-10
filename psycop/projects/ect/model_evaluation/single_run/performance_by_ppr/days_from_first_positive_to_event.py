import logging

import pandas as pd
import polars as pl

from psycop.common.cohort_definition import OutcomeTimestampFrame
from psycop.common.global_utils.mlflow.mlflow_data_extraction import EvalFrame
from psycop.common.model_training.training_output.dataclasses import (
    get_predictions_for_positive_rate,
)
from psycop.projects.cvd.model_evaluation.uuid_parsers import (
    parse_dw_ek_borger_from_uuid,
    parse_timestamp_from_uuid,
)


def days_from_first_positive_to_event(
    eval_dataset: EvalFrame,
    outcome_timestamps: OutcomeTimestampFrame,
    positive_rate: float = 0.5,
    aggregation_method: str = "sum",
) -> float:
    logging.info(f"Getting days from first positive to event with agg method {aggregation_method}")
    base_df = parse_timestamp_from_uuid(
        parse_dw_ek_borger_from_uuid(eval_dataset.frame)
    ).to_pandas()

    # Generate df with only true positives
    df = pd.DataFrame(
        {
            "id": base_df["dw_ek_borger"],
            "pred": get_predictions_for_positive_rate(
                desired_positive_rate=positive_rate, y_hat_probs=base_df["y_hat_prob"]
            )[0],
            "y": base_df["y"],
            "pred_timestamps": base_df["timestamp"],
            "outcome_timestamps": outcome_timestamps.frame.to_pandas()[
                outcome_timestamps.timestamp_col_name
            ],
        }
    )

    df = _get_time_from_first_positive_to_diagnosis_df(input_df=df)
    aggregated = df["days_from_pred_to_event"].agg(aggregation_method)
    return aggregated


def _get_time_from_first_positive_to_diagnosis_df(input_df: pd.DataFrame) -> pd.DataFrame:
    df = pl.from_pandas(input_df).with_columns(
        (pl.col("outcome_timestamps") - pl.col("pred_timestamps")).alias("time_from_pred_to_event")
    )

    ever_positives = df.filter(
        pl.col("time_from_pred_to_event").is_not_null() & pl.col("pred") == 1
    )

    plot_df = (
        ever_positives.sort("time_from_pred_to_event", descending=True)
        .groupby("id")
        .head(1)
        .with_columns(
            (pl.col("time_from_pred_to_event").dt.days() / 365.25).alias(
                "years_from_pred_to_event"
            ),
            (pl.col("time_from_pred_to_event").dt.days()).alias("days_from_pred_to_event"),
        )
    ).to_pandas()

    return plot_df
