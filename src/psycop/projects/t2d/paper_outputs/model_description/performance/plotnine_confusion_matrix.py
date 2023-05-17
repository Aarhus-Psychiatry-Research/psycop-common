import pandas as pd
import plotnine as pn
from psycop.common.model_evaluation.confusion_matrix.confusion_matrix import (
    ConfusionMatrix,
)
from psycop.common.test_utils.str_to_df import str_to_df
from psycop.projects.t2d.paper_outputs.config import PN_THEME


def plotnine_confusion_matrix(matrix: ConfusionMatrix, x_title: str) -> pn.ggplot:
    df = str_to_df(
        f"""true,pred,estimate
Yes,+,"{'{:,}'.format(matrix.true_positives)}",
Yes,-,"{'{:,}'.format(matrix.false_negatives)}",
No,+,"{'{:,}'.format(matrix.false_positives)}",
No,-,"{'{:,}'.format(matrix.true_negatives)}",
" ",+,"PPV:\n{round(matrix.ppv*100, 1)}%",
" ",-,"NPV:\n{round(matrix.npv*100,1)}%",
No," ","Spec:\n{round(matrix.specificity*100, 1)}%",
Yes," ","Sens:\n{round(matrix.sensitivity*100, 1)}%",
""",
    )

    """Create a confusion matrix and return a plotnine object."""
    df["true"] = pd.Categorical(df["true"], ["Yes", "No", " "])
    df["pred"] = pd.Categorical(df["pred"], ["+", "-", " "])

    p = (
        pn.ggplot(df, pn.aes(x="true", y="pred", fill="estimate"))
        + PN_THEME
        + pn.geom_tile(pn.aes(width=0.95, height=0.95), fill="lightgrey")
        + pn.geom_text(pn.aes(label="estimate"), size=20, color="White")
        + pn.theme(
            axis_line=pn.element_blank(),
            axis_ticks=pn.element_blank(),
            axis_text=pn.element_text(size=15),
            panel_grid_major=pn.element_blank(),
            panel_grid_minor=pn.element_blank(),
            panel_background=pn.element_blank(),
            legend_position="none",
        )
        + pn.scale_y_discrete(reverse=True)
        + pn.labs(x=x_title, y="Predicted")
    )

    return p
