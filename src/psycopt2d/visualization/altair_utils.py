from pathlib import Path
from typing import Union

import altair as alt
import pandas as pd
from altair.vegalite.v4.api import Chart


def altair_chart_to_png(
    chart: Chart,
    filename: Union[str, Path],
    height: 400,
    width: 300,
) -> Path:
    """TODO set axis labels font size.

    Args:
        chart (Chart): Altair chart
        filename (Union[str, Path]): Filename. Suffix determines the file type
        height (400): Height in pixels
        width (300): Width in pixels

    Returns:
        Path: Path to the file (filename)
    """
    filename = Path(filename)
    chart = chart.properties(height=height, width=width)
    chart.save(filename)

    return filename


if __name__ == "__main__":

    source = pd.DataFrame(
        {
            "a": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
            "b": [28, 55, 43, 91, 81, 53, 19, 87, 52],
        },
    )
    chart = alt.Chart(source).mark_bar().encode(x="a", y="b")

    chart = chart.properties(height=400, width=300)
