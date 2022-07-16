from typing import Iterable, Optional

import altair as alt
import pandas as pd


def plot_bar_chart(
    x_values: Iterable,
    y_values: Iterable,
    x_title: str,
    y_title: str,
    sort: Optional[Iterable[int]] = None,
) -> alt.Chart:
    """Plot a bar chart.

    Args:
        x_values: The x values of the bar chart.
        y_values: The y values of the bar chart.
        sort: order of values on the x-axis

    Returns:
        alt.Chart: An altair chart object.
    """
    df = pd.DataFrame({"x": x_values, "y": y_values, "sort": sort})

    if sort is not None:
        x_axis = alt.X(
            "x",
            axis=alt.Axis(title=x_title),
            sort=alt.SortField(field="sort"),
        )
    else:
        x_axis = alt.X("x", axis=alt.Axis(title=x_title))

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=x_axis,
            y=alt.Y("y", axis=alt.Axis(title=y_title)),
        )
    )
