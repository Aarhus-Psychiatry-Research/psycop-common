from typing import NewType

import pandas as pd
import polars as pl

from psycop.common.cohort_definition import OutcomeTimestampFrame, PredictionTimeFrame
from psycop.common.global_utils.cache import shared_cache
from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_time_from_first_positive_to_diagnosis_df,
)
from psycop.common.model_training.training_output.dataclasses import (
    get_predictions_for_positive_rate,
)
from psycop.projects.cvd.feature_generation.cohort_definition.cvd_cohort_definition import (
    cvd_outcome_timestamps,
    cvd_pred_times,
)
from psycop.projects.cvd.model_evaluation.single_run.sensitivity_by_time_to_event.model import (
    add_dw_ek_borger,
)
from psycop.projects.cvd.model_evaluation.single_run.single_run_artifact import RunSelector

FirstPosPredToEventDF = NewType("FirstPosPredToEventDF", pl.DataFrame)
# Contains columns "pred", "y", "id", "pred_timestamps", "outcome_timestamps"


@shared_cache.cache()
def first_positive_prediction_to_event_model(
    run: RunSelector,
    pred_timestamps: PredictionTimeFrame,
    outcome_timestamps: OutcomeTimestampFrame,
    desired_positive_rate: float = 0.05,
) -> FirstPosPredToEventDF:
    eval_df = MlflowClientWrapper().get_run(run.experiment_name, run.run_name).eval_df()

    eval_df = (
        add_dw_ek_borger(eval_df)
        .join(pred_timestamps.stripped_df, on="dw_ek_borger", suffix="_pred")
        .join(outcome_timestamps.stripped_df, on="dw_ek_borger", suffix="_outcome")
    ).to_pandas()

    df = pd.DataFrame(
        {
            "pred": get_predictions_for_positive_rate(
                desired_positive_rate=desired_positive_rate, y_hat_probs=eval_df["y_hat_prob"]
            )[0],
            "y": eval_df["y"],
            "id": eval_df["dw_ek_borger"],
            "pred_timestamps": eval_df["timestamp"],
            "outcome_timestamps": eval_df["timestamp_outcome"],
        }
    )

    plot_df = get_time_from_first_positive_to_diagnosis_df(input_df=df)

    return FirstPosPredToEventDF(pl.from_pandas(plot_df))


if __name__ == "__main__":
    test = first_positive_prediction_to_event_model(
        RunSelector(experiment_name="baseline_v2_cvd", run_name="Layer 1"),
        pred_timestamps=cvd_pred_times(),
        outcome_timestamps=cvd_outcome_timestamps(),
        desired_positive_rate=0.05,
    )

