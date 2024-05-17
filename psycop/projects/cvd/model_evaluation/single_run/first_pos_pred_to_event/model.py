from dataclasses import dataclass
from typing import NewType

import pandas as pd
import polars as pl

from psycop.common.cohort_definition import OutcomeTimestampFrame, PredictionTimeFrame
from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_time_from_first_positive_to_diagnosis_df,
)
from psycop.common.model_training.training_output.dataclasses import (
    get_predictions_for_positive_rate,
)
from psycop.projects.cvd.model_evaluation.single_run.single_run_artifact import (
    RunSelector,
    SingleRunModel,
)

FirstPosPredToEventDF = NewType("FirstPosPredToEventDF", pl.DataFrame)
# Contains columns "pred", "y", "id", "pred_timestamps", "outcome_timestamps"


@dataclass(frozen=True)
class FirstPosPredToEventModel(SingleRunModel):
    """
    Model for sensitivity by time to event.

    Args:
        desired_positive_rate: The desired positive rate.
        pred_timestamps: The prediction timestamps. Must be a dataframe with columns "timestamp" and "dw_ek_borger".
        outcome_timestamps: The outcome timestamps. Must be a dataframe with columns "timestamp" and "dw_ek_borger".
    """

    pred_timestamps: PredictionTimeFrame
    outcome_timestamps: OutcomeTimestampFrame
    desired_positive_rate: float = 0.05

    def data(self, run: RunSelector) -> FirstPosPredToEventDF:
        eval_df = MlflowClientWrapper().get_run(run.experiment_name, run.run_name).eval_df()

        eval_df = (
            eval_df.with_columns(
                pl.col("pred_time_uuid").str.split("-").first().alias("dw_ek_borger")
            )
            .join(self.pred_timestamps.stripped_df, on="dw_ek_borger", suffix="_pred")
            .join(self.outcome_timestamps.stripped_df, on="dw_ek_borger", suffix="_outcome")
        ).to_pandas()

        df = pd.DataFrame(
            {
                "pred": get_predictions_for_positive_rate(
                    desired_positive_rate=self.desired_positive_rate,
                    y_hat_probs=eval_df["y_hat_prob"],
                )[0],
                "y": eval_df["y"],
                "id": eval_df["dw_ek_borger"],
                "pred_timestamps": eval_df["timestamp_pred"],
                "outcome_timestamps": eval_df["timestamp_outcome"],
            }
        )

        plot_df = get_time_from_first_positive_to_diagnosis_df(input_df=df)

        return FirstPosPredToEventDF(pl.from_pandas(plot_df))
