from dataclasses import dataclass

import pandas as pd
import plotnine as pn

from psycop.common.model_evaluation.confusion_matrix.confusion_matrix import ConfusionMatrix
from psycop.common.test_utils.str_to_df import str_to_df
from psycop.projects.cvd.model_evaluation.single_run.confusion_matrix.model import (
    ConfusionMatrixModel,
)
from psycop.projects.cvd.model_evaluation.single_run.single_run_artifact import (
    RunSelector,
    SingleRunPlot,
)


@dataclass(frozen=True)
class ConfusionMatrixPlot(SingleRunPlot):
    data: ConfusionMatrix
    outcome_label: str
    desired_positive_rate: float = 0.05

    def __call__(self) -> pn.ggplot:
        df = str_to_df(
            f"""true,pred,estimate
        +,+,"{f'{self.data.true_positives:,}'}",
        +,-,"{f'{self.data.false_negatives:,}'}",
        -,+,"{f'{self.data.false_positives:,}'}",
        -,-,"{f'{self.data.true_negatives:,}'}",
        " ",+,"PPV:\n{round(self.data.ppv*100, 1)}%",
        " ",-,"NPV:\n{round(self.data.npv*100,1)}%",
        -," ","Spec:\n{round(self.data.specificity*100, 1)}%",
        +," ","Sens:\n{round(self.data.sensitivity*100, 1)}%",
        """
        )

        df["true"] = pd.Categorical(df["true"], ["+", "-", " "])
        df["pred"] = pd.Categorical(df["pred"], ["+", "-", " "])

        p = (
            pn.ggplot(df, pn.aes(x="true", y="pred", fill="estimate"))
            + pn.geom_tile(pn.aes(width=0.95, height=0.95), fill="gainsboro")
            + pn.geom_text(pn.aes(label="estimate"), size=18, color="black")
            + pn.theme(
                axis_line=pn.element_blank(),
                axis_ticks=pn.element_blank(),
                axis_text=pn.element_text(size=15, color="black"),
                panel_grid_major=pn.element_blank(),
                panel_grid_minor=pn.element_blank(),
                panel_background=pn.element_blank(),
                legend_position="none",
            )
            + pn.scale_y_discrete(reverse=True)
            + pn.labs(x=f"Actual {self.outcome_label}", y=f"Predicted {self.outcome_label}")
        )

        return p
