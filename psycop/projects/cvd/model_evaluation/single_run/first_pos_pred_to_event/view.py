from dataclasses import dataclass

import plotnine as pn

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.cvd.feature_generation.cohort_definition.cvd_cohort_definition import (
    cvd_outcome_timestamps,
)
from psycop.projects.cvd.model_evaluation.single_run.first_pos_pred_to_event.model import (
    FirstPosPredToEventDF,
    first_positive_prediction_to_event_model,
)
from psycop.projects.cvd.model_evaluation.single_run.single_run_artifact import SingleRunPlot
from psycop.projects.scz_bp.evaluation.configs import COLORS


@dataclass(frozen=True)
class FirstPosPredToEventPlot(SingleRunPlot):
    data: FirstPosPredToEventDF
    outcome_label: str
    desired_positive_rate: float = 0.05

    def __call__(self) -> pn.ggplot:
        plot_df = self.data.to_pandas()
        median_years = plot_df["years_from_pred_to_event"].median()

        p = (
            pn.ggplot(plot_df, pn.aes(x="years_from_pred_to_event", fill="y"))  # type: ignore
            + pn.geom_density(alpha=0.8, fill="#0072B2")
            + pn.xlab("Years from first positive prediction\n to event")
            + pn.scale_x_reverse(breaks=range(int(plot_df["years_from_pred_to_event"].max() + 1)))
            + pn.ylab("Proportion")
            + pn.geom_vline(xintercept=median_years, linetype="dashed", size=1)
            + pn.geom_vline(xintercept=0, linetype="solid", size=1)
        )
        p.save("test.png")
        return p


if __name__ == "__main__":
    eval_frame = (
        MlflowClientWrapper()
        .get_run(experiment_name="baseline_v2_cvd", run_name="Layer 1")
        .eval_frame()
    )
    outcome_timestamps = cvd_outcome_timestamps()

    plot = FirstPosPredToEventPlot(
        data=first_positive_prediction_to_event_model(
            eval_df=eval_frame.frame, outcome_timestamps=outcome_timestamps
        ),
        outcome_label="CVD",
    )()
    plot.save("test.png")

    pass
