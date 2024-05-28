from typing import NewType

import pandas as pd
import polars as pl

from psycop.common.cohort_definition import OutcomeTimestampFrame, PredictionTimeFrame
from psycop.common.global_utils.cache import shared_cache
from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_time_from_first_positive_to_diagnosis_df,
)
from psycop.common.model_training.training_output.dataclasses import (
    get_predictions_for_positive_rate,
)
from psycop.projects.cvd.model_evaluation.uuid_parsers import (
    parse_dw_ek_borger_from_uuid,
    parse_timestamp_from_uuid,
)

FirstPosPredToEventDF = NewType("FirstPosPredToEventDF", pl.DataFrame)
# Contains columns "pred", "y", "id", "pred_timestamps", "outcome_timestamps"


@shared_cache.cache()
def first_positive_prediction_to_event_model(
    eval_df: pl.DataFrame,
    outcome_timestamps: OutcomeTimestampFrame,
    desired_positive_rate: float = 0.05,
) -> FirstPosPredToEventDF:
    eval_dataset = (
        parse_timestamp_from_uuid(parse_dw_ek_borger_from_uuid(eval_df)).join(
            outcome_timestamps.essentials_df, on="dw_ek_borger", suffix="_outcome"
        )
    ).to_pandas()

    df = pd.DataFrame(
        {
            "pred": get_predictions_for_positive_rate(
                desired_positive_rate=desired_positive_rate, y_hat_probs=eval_dataset["y_hat_prob"]
            )[0],
            "y": eval_dataset["y"],
            "id": eval_dataset["dw_ek_borger"],
            "pred_timestamps": eval_dataset["timestamp"],
            "outcome_timestamps": eval_dataset["timestamp_outcome"],
        }
    )

    plot_df = get_time_from_first_positive_to_diagnosis_df(input_df=df)

    return FirstPosPredToEventDF(pl.from_pandas(plot_df))
