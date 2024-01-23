import pandas as pd
import plotnine as pn

from psycop.common.model_evaluation.confusion_matrix.confusion_matrix import ConfusionMatrix
from psycop.common.test_utils.str_to_df import str_to_df
from psycop.projects.t2d.paper_outputs.config import T2D_PN_THEME


def plotnine_confusion_matrix(matrix: ConfusionMatrix, outcome_text: str) -> pn.ggplot:
    df = str_to_df(
        f"""true,pred,estimate
+,+,"{f'{matrix.true_positives:,}'}",
+,-,"{f'{matrix.false_negatives:,}'}",
-,+,"{f'{matrix.false_positives:,}'}",
-,-,"{f'{matrix.true_negatives:,}'}",
" ",+,"PPV:\n{round(matrix.ppv*100, 1)}%",
" ",-,"NPV:\n{round(matrix.npv*100,1)}%",
-," ","Spec:\n{round(matrix.specificity*100, 1)}%",
+," ","Sens:\n{round(matrix.sensitivity*100, 1)}%",
"""
    )

    """Create a confusion matrix and return a plotnine object."""
    df["true"] = pd.Categorical(df["true"], ["+", "-", " "])
    df["pred"] = pd.Categorical(df["pred"], ["+", "-", " "])

    p = (
        pn.ggplot(df, pn.aes(x="true", y="pred", fill="estimate"))
        + T2D_PN_THEME
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
        + pn.labs(x=f"Actual {outcome_text}", y=f"Predicted {outcome_text}")
    )

    return p
