import pandas as pd
import plotnine as pn

from psycop.projects.scz_bp.evaluation.configs import Colors
from psycop.projects.t2d.paper_outputs.config import COLORS, FONT_SIZES, THEME


def auroc_by_view(df: pd.DataFrame, x_column: str, line_y_col_name: str, xlab: str) -> pn.ggplot:
    """Plot robustness of model performance by a given input variable."""
    df["proportion_of_n"] = df["n_in_bin"] / df["n_in_bin"].sum()

    p = (
        pn.ggplot(df, pn.aes(x=x_column, y=line_y_col_name))
        + pn.geom_bar(
            pn.aes(x=x_column, y="proportion_of_n"), stat="identity", fill=COLORS.background
        )
        + pn.geom_path(group=1, size=1)
        + pn.xlab(xlab)
        + pn.ylab("AUROC / Proportion of patients")
        + pn.ylim(0, 1)
        + THEME
        + pn.theme(
            axis_text_x=pn.element_text(size=FONT_SIZES.axis_tick_labels, angle=45, hjust=1),
            axis_text_y=pn.element_text(size=FONT_SIZES.axis_tick_labels),
        )
    )

    if "ci_lower" in df.columns:
        p += pn.geom_point(size=1)
        p += pn.geom_errorbar(pn.aes(ymin="ci_lower", ymax="ci_upper"), width=0.1)

    if df[x_column].nunique() < 3:
        p += pn.scale_x_discrete()

    return p
