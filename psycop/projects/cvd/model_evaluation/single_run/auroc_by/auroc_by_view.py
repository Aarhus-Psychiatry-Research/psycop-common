import pandas as pd
import plotnine as pn

from psycop.projects.t2d.paper_outputs.config import COLORS


def auroc_by_view(
    df: pd.DataFrame,
    x_column: str,
    line_y_col_name: str,
    xlab: str,
    rotate_x_axis_labels_degrees: int = 45,
) -> pn.ggplot:
    """Plot robustness of model performance by a given input variable.

    Args:
        df (pd.DataFrame): Dataframe
        x_column (str): Column name for x-axis values
        line_y_col_name (str): Column name for y-axis values for line
        xlab (str): x-axis label
        figure_file_name (str): Filename for output figure (without extension)
        rotate_x_axis_labels_degrees (int, optional): Degrees to rotate x-axis labels. Defaults to 45.
    """
    df["proportion_of_n"] = df["n_in_bin"] / df["n_in_bin"].sum()

    p = (
        pn.ggplot(df, pn.aes(x=x_column, y=line_y_col_name))
        + pn.geom_bar(
            pn.aes(x=x_column, y="proportion_of_n"), stat="identity", fill=COLORS.background
        )
        + pn.geom_path(group=1, color=COLORS.primary, size=1)
        + pn.xlab(xlab)
        + pn.ylab("AUROC / Proportion of patients")
        + pn.theme(axis_text_x=pn.element_text(angle=rotate_x_axis_labels_degrees, hjust=1))
        + pn.ylim(0, 1)
    )

    if "ci_lower" in df.columns:
        p += pn.geom_pointrange(
            pn.aes(ymin="ci_lower", ymax="ci_upper"), color=COLORS.primary, size=0.5
        )

    if df[x_column].nunique() < 3:
        p += pn.scale_x_discrete()

    # Draw the plot to check that it works
    p.draw()

    return p
