"""Base charts."""
from collections.abc import Iterable
from typing import Optional

import altair as alt
import pandas as pd


def plot_bar_chart(
    x_values: Iterable,
    y_values: Iterable,
    x_title: str,
    y_title: str,
    sort_x: Optional[Iterable[int]] = None,
    sort_y: Optional[Iterable[int]] = None,
) -> alt.Chart:
    """Plot a basic bar chart with Altair.

    Args:
        x_values (Iterable): The x values of the bar chart.
        y_values (Iterable): The y values of the bar chart.
        x_title (str): title of x axis
        y_title (str): title of y axis
        sort_x (Optional[Iterable[int]], optional): order of values on the x-axis. Defaults to None.
        sort_y (Optional[Iterable[int]], optional): order of values on the y-axis. Defaults to None.

    Returns:
        alt.Chart: Altair chart
    """

    df = pd.DataFrame({"x": x_values, "y": y_values, "sort": sort_x})

    if sort_x is not None:
        x_axis = alt.X(
            "x",
            axis=alt.Axis(title=x_title),
            sort=alt.SortField(field="sort"),
        )
    else:
        x_axis = alt.X("x", axis=alt.Axis(title=x_title))

    if sort_y is not None:
        y_axis = alt.Y(
            "y",
            axis=alt.Axis(title=y_title),
            sort=alt.SortField(field="sort"),
        )
    else:
        y_axis = alt.Y("y", axis=alt.Axis(title=y_title))

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=x_axis,
            y=y_axis,
        )
    )
