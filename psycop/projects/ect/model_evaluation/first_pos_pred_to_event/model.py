from typing import NewType

import pandas as pd
import polars as pl

from psycop.common.global_utils.cache import shared_cache
from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_time_from_first_positive_to_diagnosis_df,
)
from psycop.common.model_training.training_output.dataclasses import (
    get_predictions_for_positive_rate,
)

FirstPosPredToEventDF = NewType("FirstPosPredToEventDF", pl.DataFrame)
# Contains columns "pred", "y", "id", "pred_timestamps", "outcome_timestamps"


@shared_cache().cache()
def first_positive_prediction_to_event_model(
    eval_df: pl.DataFrame, desired_positive_rate: float = 0.02
) -> FirstPosPredToEventDF:
    eval_dataset = eval_df.to_pandas()

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
