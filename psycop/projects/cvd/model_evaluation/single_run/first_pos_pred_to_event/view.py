from dataclasses import dataclass

import pandas as pd
import plotnine as pn

from psycop.common.model_evaluation.confusion_matrix.confusion_matrix import ConfusionMatrix
from psycop.common.test_utils.str_to_df import str_to_df
from psycop.projects.cvd.model_evaluation.single_run.confusion_matrix.model import (
    ConfusionMatrixModel,
)
from psycop.projects.cvd.model_evaluation.single_run.first_pos_pred_to_event.model import (
    FirstPosPredToEventDF,
)
from psycop.projects.cvd.model_evaluation.single_run.single_run_artifact import (
    RunSelector,
    SingleRunPlot,
)
from psycop.projects.restraint.model_evaluation.figures.roc_curve import plot_auc_roc


@dataclass(frozen=True)
class FirstPosPredToEventPlot(SingleRunPlot):
    data: FirstPosPredToEventDF
    outcome_label: str
    desired_positive_rate: float = 0.05

    def __call__(self) -> pn.ggplot:
        plot_df = self.data.to_pandas()
        median_years = plot_df["years_from_pred_to_event"].median()
        annotation_text = f"Median: {round(median_years, 1)!s} years"

        p = (
            pn.ggplot(plot_df, pn.aes(x="years_from_pred_to_event"))  # type: ignore
            + pn.geom_histogram(binwidth=1, fill="orange")
            + pn.xlab("Years from first positive prediction\n to event")
            + pn.scale_x_reverse(breaks=range(int(plot_df["years_from_pred_to_event"].max() + 1)))
            + pn.ylab("n")
            + pn.geom_vline(xintercept=median_years, linetype="dashed", size=1)
            + pn.geom_text(
                pn.aes(x=median_years, y=40),
                label=annotation_text,
                ha="right",
                nudge_x=-0.3,
                size=11,
            )
        )
        return p
