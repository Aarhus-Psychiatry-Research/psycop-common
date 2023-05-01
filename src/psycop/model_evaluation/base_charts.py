"""Base charts."""
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series


def plot_basic_chart(
    x_values: Series,
    y_values: Union[Series, Sequence[Series]],
    x_title: str,
    y_title: str,
    plot_type: Union[list[str], str],
    labels: Optional[list[str]] = None,
    sort_x: Optional[Sequence[int]] = None,
    sort_y: Optional[Sequence[int]] = None,
    confidence_interval: Optional[Sequence[tuple[float, float]]] = None,
    flip_x_axis: bool = False,
    flip_y_axis: bool = False,
    bar_count_values: Optional[pd.Series] = None,
    bar_count_y_axis_title: str = "Number of observations",
    y_limits: Optional[tuple[float, float]] = None,
    fig_size: Optional[tuple[float, float]] = (5, 5),
    dpi: Optional[int] = 300,
    save_path: Optional[Union[Path, str]] = None,
) -> Union[None, Path]:
    """Plot a simple chart using matplotlib. Options for sorting the x and y
    axis are available.

    Args:
        x_values: The x values of the bar chart.
        y_values: The y values of the bar chart.
        x_title: title of x axis
        y_title: title of y axis
        plot_type: type of plots.
            Options are combinations of ["bar", "hbar", "line", "scatter"] Defaults to "bar".
        labels: Optional labels to add to the plot(s).
        sort_x: order of values on the x-axis. Defaults to None.
        sort_y: order of values on the y-axis. Defaults to None.
        confidence_interval: Confidence interval for plotting. Defaults to None. If None, no confidence interval is plotted.
            If you supple a sequence of series for the y_values, it will only plot the confidence interval for the last series.
        save_path: path to save figure. Defaults to None.
        flip_x_axis: Whether to flip the x axis. Defaults to False.
        flip_y_axis: Whether to flip the y axis. Defaults to False.
        bar_count_values: Values to use for overlaid histogram of n in bins. Defaults to None.
        bar_count_y_axis_title: Title of y axis of overlaid histogram. Defaults to "Number of observations".
        y_limits: y-axis limits. Defaults to None.
        fig_size: figure size. Defaults to None.
        dpi: dpi of figure. Defaults to 300.
        save_path: Path to save figure. Defaults to None.

    Returns:
        Union[None, Path]: None if save_path is None, else path to saved figure
    """
    if isinstance(plot_type, str):
        plot_type = [plot_type]

    df = pd.DataFrame(
        {"x": x_values, "sort_x": sort_x, "sort_y": sort_y},
    )

    if sort_x is not None:
        df = df.sort_values(by=["sort_x"])

    if sort_y is not None:
        df = df.sort_values(by=["sort_y"])

    fig = plt.figure(figsize=fig_size, dpi=dpi)
    axs = fig.subplots()

    y_sequences: list[Series] = [y_values] if not isinstance(y_values[0], pd.Series) else y_values  # type: ignore

    plot_functions = {
        "bar": axs.bar,  # type: ignore
        "hbar": axs.barh,  # type: ignore
        "line": axs.plot,  # type: ignore
        "scatter": axs.scatter,  # type: ignore
    }

    # choose the first plot type as the one to use for legend
    legend_plot = plot_type[0]

    label_plots = []
    for y_series in y_sequences:
        for p_type in plot_type:
            plot_function: Callable = plot_functions.get(p_type)  # type: ignore

            color = "orange" if p_type == "bar" else None
            plot = plot_function(df["x"], y_series, color=color)

            if p_type == legend_plot:
                # need to one of the plot types for labelling
                label_plots.append(plot)
            if p_type == "hbar":
                plt.yticks(fontsize=7)

    # # add error bars
    if confidence_interval is not None:
        ci_matrix = np.array(confidence_interval)
        # convert lower and upper bound to relative difference
        # as matplotlib errorbar interpret yerr as an offset
        ci_matrix = np.abs(ci_matrix - y_series.to_numpy().reshape(-1, 1))  # type: ignore
        plt.errorbar(x=df["x"], y=y_series, yerr=ci_matrix.T, fmt="none", capsize=5)  # type: ignore

    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.xticks(fontsize=7)
    plt.xticks(rotation=45)

    if y_limits is not None:
        plt.ylim(y_limits)

    if flip_x_axis:
        plt.gca().invert_xaxis()
    if flip_y_axis:
        plt.gca().invert_yaxis()
    if labels is not None:
        plt.gca().legend(
            [plot[0] for plot in label_plots],
            [str(label) for label in labels],
            loc="upper left",
            bbox_to_anchor=(0.14, 0.95),
            frameon=True,
        )

    if bar_count_values is not None:
        # add additional y-axis for count
        bar_overlay = plt.gca().twinx()
        bar_overlay.bar(df["x"], bar_count_values, color="gainsboro", alpha=0.5)
        bar_overlay.set_ylabel(bar_count_y_axis_title)

        # put bar plots behind other plots
        axs.set_zorder(bar_overlay.get_zorder() + 1)  # type: ignore
        axs.set_facecolor("none")  # type: ignore
        bar_overlay.set_facecolor("none")

    plt.tight_layout()

    if save_path is not None:
        print(f"Saving to {save_path}")
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path)

    plt.close()

    return save_path
