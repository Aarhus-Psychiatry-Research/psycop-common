from dataclasses import dataclass

import plotnine as pn

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.scz_bp.evaluation.configs import Colors
from psycop.projects.t2d.paper_outputs.config import THEME
from psycop.projects.t2d_bigdata.feature_generation.cohort_definition.t2d_bigdata_cohort_definer import (
    t2d_bigdata_outcome_timestamps,
)
from psycop.projects.t2d_bigdata.model_evaluation.single_run.first_pos_pred_to_event.model import (
    FirstPosPredToEventDF,
    first_positive_prediction_to_event_model,
)
from psycop.projects.t2d_bigdata.model_evaluation.single_run.single_run_artifact import (
    SingleRunPlot,
)


@dataclass(frozen=True)
class FirstPosPredToEventPlot(SingleRunPlot):
    data: FirstPosPredToEventDF
    outcome_label: str
    desired_positive_rate: float = 0.05
    fill_color: str = Colors.primary

    def __call__(self) -> pn.ggplot:
        plot_df = self.data.to_pandas()
        median_years = plot_df["years_from_pred_to_event"].median()

        p = (
            pn.ggplot(plot_df, pn.aes(x="years_from_pred_to_event", fill="y"))  # type: ignore
            + pn.geom_density(alpha=0.8, fill=self.fill_color)
            + pn.xlab("Years until event")
            + pn.scale_x_continuous(
                breaks=range(int(plot_df["years_from_pred_to_event"].max() + 1))
            )
            + pn.ylab("Proportion")
            + pn.geom_vline(xintercept=median_years, linetype="dashed", size=1)
            + THEME
        )
        p.save("test.png")
        return p


if __name__ == "__main__":
    eval_frame = (
        MlflowClientWrapper()
        .get_run(experiment_name="T2D-bigdata", run_name="Layer 1")
        .eval_frame()
    )
    outcome_timestamps = t2d_bigdata_outcome_timestamps()

    plot = FirstPosPredToEventPlot(
        data=first_positive_prediction_to_event_model(
            eval_df=eval_frame.frame, outcome_timestamps=outcome_timestamps
        ),
        outcome_label="T2D-bigdata",
    )()
    plot.save("test.png")
