from pathlib import Path
from typing import Union

import wandb
from altair.vegalite.v4.api import Chart
from wandb.sdk.wandb_run import Run as wandb_run


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
        filepath (Union[str, Path]): Filepath including suffix. Suffix determines the file type.
        height (int): Height in pixels
        width (int): Width in pixels
        font_size (int): How large the axis labels should be

    Returns:
        str: Path to the file (filename)
    """
    filepath = Path(filepath)
    chart = chart.properties(height=height, width=width)
    chart = chart.configure_axis(labelFontSize=font_size, titleFontSize=font_size)
    chart.save(filepath)

    return str(filepath)


def log_altair_to_wandb(chart: Chart, chart_name: str, run: wandb_run):  # type: ignore
    """Helper to log Altair charts.

    Args:
        chart (Chart): Altair chart
        chart_name (str): Name of the chart
        run (wandb.run): Wandb run object
    """
    base_path = Path(__file__).parent.parent.parent.parent / ".tmp_figures"

    # Create folder if it doesn't exist
    Path.mkdir(base_path, parents=True, exist_ok=True)

    plot_html = altair_chart_to_file(
        chart=chart,
        filepath=base_path / f"{chart_name}.html",
    )

    run.log({f"html_{chart_name}": wandb.Html(plot_html)})


def log_altair_to_disk(chart: Chart, chart_name: str, path: Path):
    """Log Altair chart to disk.

    Args:
        chart (Chart): Altair chart
        chart_name (str): Name of the chart
        path (Path): Path to the folder where the chart should be saved
    """
    # Create folder if it doesn't exist
    Path.mkdir(path, parents=True, exist_ok=True)

    altair_chart_to_file(
        chart=chart,
        filepath=path / f"{chart_name}.png",
    )
