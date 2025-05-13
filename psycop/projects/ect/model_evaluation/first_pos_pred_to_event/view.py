from dataclasses import dataclass
from pathlib import Path

import plotnine as pn

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.ect.feature_generation.cohort_definition.ect_cohort_definition import (
    ect_outcome_timestamps,
)
from psycop.projects.ect.feature_generation.cohort_definition.outcome_specification.combined import (
    add_first_ect_time_after_prediction_time,
)
from psycop.projects.ect.model_evaluation.first_pos_pred_to_event.model import (
    FirstPosPredToEventDF,
    first_positive_prediction_to_event_model,
)
from psycop.projects.ect.model_evaluation.single_run_artifact import SingleRunPlot
from psycop.projects.scz_bp.evaluation.configs import Colors
from psycop.projects.t2d.paper_outputs.config import THEME


@dataclass(frozen=True)
class FirstPosPredToEventPlot(SingleRunPlot):
    data: FirstPosPredToEventDF
    outcome_label: str
    desired_positive_rate: float = 0.05
    fill_color: str = Colors.primary
    max_days: int = 100

    def __call__(self) -> pn.ggplot:
        plot_df = self.data.to_pandas()
        median_days = plot_df["days_from_pred_to_event"].median()
        annotation_text = f"Median: {round(median_days, 1)!s} days"

        p = (
            pn.ggplot(plot_df, pn.aes(x="days_from_pred_to_event", fill="y"))  # type: ignore
            + pn.geom_density(alpha=0.8, fill=self.fill_color)
            + pn.xlab("Days from first positive prediction to event")
            + pn.xlim((0, self.max_days))
            + pn.ylab("Proportion")
            + pn.geom_vline(xintercept=median_days, linetype="dashed", size=1)
            + pn.geom_text(
                pn.aes(x=median_days + 5, y=0.025),
                label=annotation_text,
                ha="left",
                nudge_x=-0.3,
                size=11,
            )
            + THEME
        )
        return p


if __name__ == "__main__":
    eval_frame = (
        MlflowClientWrapper()
        .get_run(
            experiment_name="ECT hparam, structured_only, xgboost, no lookbehind filter",
            run_name="inquisitive-koi-243",
        )
        .eval_frame()
    )
    outcome_timestamps = ect_outcome_timestamps()

    # add "outcome_timestamp" column with time of the first ECT after the prediction time
    eval_df = add_first_ect_time_after_prediction_time(eval_frame.frame)

    plot = FirstPosPredToEventPlot(
        data=first_positive_prediction_to_event_model(eval_df=eval_df), outcome_label="ECT"
    )()
    plot.save(Path(__file__).parent / "test.png")
