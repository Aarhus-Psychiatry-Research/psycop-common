from typing import Iterable

import altair as alt
import pandas as pd


def plot_bar_chart(
    x_values: Iterable,
    y_values: Iterable,
    x_title: str,
    y_title: str,
) -> alt.Chart:
    """Plot a bar chart.

    Args:
        x_values: The x values of the bar chart.
        y_values: The y values of the bar chart.

    Returns:
        alt.Chart: An altair chart object.
    """
    return (
        alt.Chart(pd.DataFrame({"x": x_values, "y": y_values}))
        .mark_bar()
        .encode(
            x=alt.X("x", axis=alt.Axis(title=x_title)),
            y=alt.Y("y", axis=alt.Axis(title=y_title)),
        )
    )
