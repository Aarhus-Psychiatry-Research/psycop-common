from pathlib import Path
from typing import Union

from altair.vegalite.v4.api import Chart


def altair_chart_to_png(
    chart: Chart,
    filename: Union[str, Path],
    height: int = 400,
    width: int = 300,
    font_size: int = 18,
) -> Path:
    """Helper to save Altair charts.

    Args:
        chart (Chart): Altair chart
        filename (Union[str, Path]): Filename. Suffix determines the file type
        height (int): Height in pixels
        width (int): Width in pixels
        font_size (int): How large the axis labels should be

    Returns:
        Path: Path to the file (filename)
    """
    filename = Path(filename)
    chart = chart.properties(height=height, width=width)
    chart = chart.configure_axis(labelFontSize=font_size, titleFontSize=font_size)
    chart.save(filename)

    return filename
