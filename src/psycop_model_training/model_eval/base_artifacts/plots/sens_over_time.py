"""Generate a plot of sensitivity by time to outcome."""
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Literal, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from psycop_model_training.model_eval.dataclasses import EvalDataset
from psycop_model_training.utils.utils import round_floats_to_edge


def create_sensitivity_by_time_to_outcome_df(
    eval_dataset: EvalDataset,
    desired_positive_rate: float,
    outcome_timestamps: Iterable[pd.Timestamp],
    prediction_timestamps: Iterable[pd.Timestamp],
    bins: Iterable = (0, 1, 7, 14, 28, 182, 365, 730, 1825),
    bin_delta: Literal["D", "W", "M", "Q", "Y"] = "D",
) -> pd.DataFrame:
    """Calculate sensitivity by time to outcome.

    Args:
        eval_dataset (EvalDataset): Eval dataset.
        desired_positive_rate (float): Takes the top positive_rate% of predicted probabilities and turns them into 1, the rest 0.
        outcome_timestamps (Iterable[pd.Timestamp]): Timestamp of the outcome, if any.
        prediction_timestamps (Iterable[pd.Timestamp]): Timestamp of the prediction.
        bins (list, optional): Default bins for time to outcome. Defaults to [0, 1, 7, 14, 28, 182, 365, 730, 1825].
        bin_delta (str, optional): The unit of time for the bins. Defaults to "D".

    Returns:
        pd.DataFrame
    """

    y_hat_series, actual_positive_rate = eval_dataset.get_predictions_for_positive_rate(
        desired_positive_rate=desired_positive_rate,
    )

    df = pd.DataFrame(
        {
            "y": eval_dataset.y,
            "y_hat": y_hat_series,
            "outcome_timestamp": outcome_timestamps,
            "prediction_timestamp": prediction_timestamps,
        },
    )

    # Get proportion of y_hat == 1, which is equal to the actual positive rate in the data.
    threshold_percentile = round(
        actual_positive_rate * 100,
        2,
    )

    df = df[df["y"] == 1]

    # Calculate difference in days between columns
    df["days_to_outcome"] = (
        df["outcome_timestamp"] - df["prediction_timestamp"]
    ) / np.timedelta64(
        1,
        bin_delta,
    )  # type: ignore

    df["true_positive"] = (df["y"] == 1) & (df["y_hat"] == 1)
    df["false_negative"] = (df["y"] == 1) & (df["y_hat"] == 0)

    df["days_to_outcome_binned"] = round_floats_to_edge(
        df["days_to_outcome"],
        bins=bins,
    )

    output_df = (
        df[["days_to_outcome_binned", "true_positive", "false_negative"]]
        .groupby("days_to_outcome_binned")
        .sum()
    )

    output_df["sens"] = round(
        output_df["true_positive"]
        / (output_df["true_positive"] + output_df["false_negative"]),
        2,
    )

    # Prep for plotting
    ## Save the threshold for each bin
    output_df["threshold"] = desired_positive_rate

    output_df["threshold_percentile"] = threshold_percentile

    output_df = output_df.reset_index()

    # Convert days_to_outcome_binned to string for plotting
    output_df["days_to_outcome_binned"] = output_df["days_to_outcome_binned"].astype(
        str,
    )

    return output_df


def _generate_sensitivity_array(
    df: pd.DataFrame,
    n_decimals_y_axis: int,
) -> tuple[np.ndarray, list[str], list[float]]:
    """Generate sensitivity array for plotting heatmap.

    Args:
        df (pd.DataFrame): Dataframe with columns "sens", "days_to_outcome_binned" and "threshold".
        n_decimals_y_axis (int): Number of decimals to round y axis labels to.

    Returns:
        A tuple containing the generated sensitivity array (np.ndarray), the x axis labels and the y axis labels rounded to n_decimals_y_axis.
    """
    x_labels = df["days_to_outcome_binned"].unique().tolist()

    y_labels = df["threshold_percentile"].unique().tolist()

    y_labels_rounded = [
        round(y_labels[value], n_decimals_y_axis) for value in range(len(y_labels))
    ]

    sensitivity_array = []

    for threshold in df["threshold"].unique().tolist():
        sensitivity_current_threshold = []

        df_subset_y = df[df["threshold"] == threshold]

        for days_interval in x_labels:
            df_subset_y_x = df_subset_y[
                df_subset_y["days_to_outcome_binned"] == days_interval
            ]
            if len(df_subset_y_x["sens"].unique().tolist()) > 1:
                raise ValueError(
                    f"More than one sensitivity value for this threshold ({threshold}) and days interval ({days_interval}).",
                )
            sensitivity_current_threshold.append(
                df_subset_y_x["sens"].unique().tolist()[0],
            )

        sensitivity_array.append(sensitivity_current_threshold)

    return (
        np.array(sensitivity_array),
        x_labels,
        y_labels_rounded,
    )


def _annotate_heatmap(
    image: mpl.image.AxesImage,
    data: Optional[np.ndarray] = None,
    textcolors: tuple = ("black", "white"),
    threshold: Optional[float] = None,
    **textkw: dict,
) -> list:
    """A function to annotate a heatmap. Adapted from mpl documentation.

    Args:
        image (mpl.image.AxesImage): The AxesImage to be labeled.
        data (np.ndarray): Data used to annotate. If None, the image's data is used. Defaults to None.
        textcolors (tuple, optional): A pair of colors. The first is used for values below a threshold, the second for those above. Defaults to ("black", "white").
        threshold (float, optional): Value in data units according to which the colors from textcolors are applied. If None (the default) uses the middle of the colormap as separation. Defaults to None.
        **textkw (dict, optional): All other arguments are forwarded to each call to `text` used to create the text labels. Defaults to {}.

    Returns:
        texts (list): A list of mpl.text.Text instances for each label.
    """

    if not isinstance(data, (list, np.ndarray)):
        data: np.ndarray = image.get_array()  # type: ignore

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = image.norm(threshold)
    else:
        threshold = image.norm(data.max()) / 2.0  # type: ignore

    test_kwargs = {
        "horizontalalignment": "center",
        "verticalalignment": "center",
    } | textkw

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []

    for heat_row_idx in range(data.shape[0]):  # type: ignore
        for heat_col_idx in range(data.shape[1]):  # type: ignore
            test_kwargs.update(
                color=textcolors[
                    int(image.norm(data[heat_row_idx, heat_col_idx]) > threshold)  # type: ignore
                ],
            )
            text = image.axes.text(
                heat_col_idx,
                heat_row_idx,
                str(data[heat_row_idx, heat_col_idx]),  # type: ignore
                **test_kwargs,
            )
            texts.append(text)

    return texts


def _format_sens_by_time_heatmap(
    colorbar_label: str,
    x_title: str,
    y_title: str,
    data: np.ndarray,
    x_labels: list[str],
    y_labels: list[float],
    fig: plt.Figure,
    axes: plt.Axes,
    image: mpl.image.AxesImage,
) -> tuple[plt.Figure, plt.Axes]:
    # Create colorbar
    cbar = axes.figure.colorbar(image, ax=axes)
    cbar.ax.set_ylabel(colorbar_label, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    axes.set_xticks(np.arange(data.shape[1]), labels=x_labels)
    axes.set_yticks(np.arange(data.shape[0]), labels=y_labels)

    # Let the horizontal axes labeling appear on top.
    axes.tick_params(
        top=False,
        bottom=True,
        labeltop=False,
        labelbottom=True,
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(
        axes.get_xticklabels(),
        rotation=90,
        ha="right",
        rotation_mode="anchor",
    )

    # Turn spines off and create white grid.
    axes.spines[:].set_visible(False)

    axes.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    axes.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    axes.grid(which="minor", color="w", linestyle="-", linewidth=3)
    axes.tick_params(which="minor", bottom=False, left=False)

    # Add annotations
    _annotate_heatmap(image)

    # Set axis labels and title
    axes.set(
        xlabel=x_title,
        ylabel=y_title,
    )

    fig.tight_layout()

    return fig, axes
