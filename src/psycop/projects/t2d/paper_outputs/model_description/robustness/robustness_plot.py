import pandas as pd
import plotnine as pn
from psycop.projects.t2d.paper_outputs.config import COLORS, PN_THEME


def plot_robustness(
    df: pd.DataFrame,
    x_column: str,
    line_y_col_name: str,
    bar_y_col_name: str,
    xlab: str,
    ylab: str,
) -> pn.ggplot:
    """Plot robustness of model performance by a given input variable.

    Args:
        df (pd.DataFrame): Dataframe
        x_column (str): Column name for x-axis values
        line_y_col_name (str): Column name for y-axis values for line
        bar_y_col_name (str): Column name for y-axis values for bar
        xlab (str): x-axis label
        ylab (str): y-axis label
    """
    p = (
        pn.ggplot(df, pn.aes(x=x_column, y=line_y_col_name))
        + pn.geom_bar(
            pn.aes(x=x_column, y=bar_y_col_name),
            stat="identity",
            fill=COLORS.background,
        )
        + pn.geom_path(group=1, color=COLORS.primary, size=1)
        + pn.xlab(xlab)
        + pn.ylab(ylab)
        + PN_THEME
    )

    if "ci_lower" in df.columns:
        p += pn.geom_pointrange(
            pn.aes(ymin="ci_lower", ymax="ci_upper"),
            color=COLORS.primary,
            size=0.5,
        )

    return p
