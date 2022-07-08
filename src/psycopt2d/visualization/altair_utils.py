from pathlib import Path
from typing import Union

from altair.vegalite.v4.api import Chart

import wandb


def altair_chart_to_file(
    chart: Chart,
    filepath: Union[str, Path],
    height: int = 400,
    width: int = 300,
    font_size: int = 18,
) -> str:
    """Helper to save Altair charts.

    Args:
        chart (Chart): Altair chart
        filename (Union[str, Path]): Filename including suffix. Suffix determines the file type.
        height (int): Height in pixels
        width (int): Width in pixels
        font_size (int): How large the axis labels should be

    Returns:
        Path: Path to the file (filename)
    """
    filepath = Path(filepath)
    chart = chart.properties(height=height, width=width)
    chart = chart.configure_axis(labelFontSize=font_size, titleFontSize=font_size)
    chart.save(filepath)

    return str(filepath)


def log_altair_to_vega_and_png(chart: Chart, chart_name: str, run: wandb.run):
    """Helper to log Altair charts.

    Args:
        chart (Chart): Altair chart
        chart_name (str): Name of the chart
        run (wandb.run): Wandb run

    Returns:
        None
    """
    base_path = Path(__file__).parent.parent.parent.parent / "outputs" / "figures"

    # plot_png_path = altair_chart_to_file(
    #     chart=chart,
    #     filepath=base_path / f"{chart_name}.png",
    # )
    # run.log({f"png_{chart_name}": wandb.Image(plot_png_path)})

    plot_html = altair_chart_to_file(
        chart=chart,
        filepath=base_path / f"{chart_name}.html",
    )

    run.log({f"html_{chart_name}": wandb.Html(plot_html)})
