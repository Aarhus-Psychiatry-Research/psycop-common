"""Generate a plot of sensitivity by time to outcome."""
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from psycop_model_training.model_eval.dataclasses import EvalDataset
from psycop_model_training.utils.utils import round_floats_to_edge


def create_sensitivity_by_time_to_outcome_df(
    labels: Iterable[int],
    y_hat_probs: Iterable[int],
    pred_proba_threshold: float,
    outcome_timestamps: Iterable[pd.Timestamp],
    prediction_timestamps: Iterable[pd.Timestamp],
    bins: Iterable = (0, 1, 7, 14, 28, 182, 365, 730, 1825),
) -> pd.DataFrame:
    """Calculate sensitivity by time to outcome.

    Args:
        labels (Iterable[int]): True labels of the data.
        y_hat_probs (Iterable[int]): Predicted label probability.
        pred_proba_threshold (float): The pred_proba threshold above which predictions are classified as positive.
        outcome_timestamps (Iterable[pd.Timestamp]): Timestamp of the outcome, if any.
        prediction_timestamps (Iterable[pd.Timestamp]): Timestamp of the prediction.
        bins (list, optional): Default bins for time to outcome. Defaults to [0, 1, 7, 14, 28, 182, 365, 730, 1825].

    Returns:
        pd.DataFrame
    """

    # Modify pandas series to 1 if y_hat is larger than threshold, otherwise 0
    y_hat = pd.Series(y_hat_probs).apply(
        lambda x: 1 if x > pred_proba_threshold else 0,
    )

    df = pd.DataFrame(
        {
            "y": labels,
            "y_hat": y_hat,
            "outcome_timestamp": outcome_timestamps,
            "prediction_timestamp": prediction_timestamps,
        },
    )

    # Get proportion of y_hat == 1, which is equal to the positive rate in the data
    threshold_percentile = round(
        df[df["y_hat"] == 1].shape[0] / df.shape[0] * 100,
        2,
    )

    df = df[df["y"] == 1]

    # Calculate difference in days between columns
    df["days_to_outcome"] = (
        df["outcome_timestamp"] - df["prediction_timestamp"]
    ) / np.timedelta64(
        1,
        "D",
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
    output_df["threshold"] = pred_proba_threshold

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
):
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
    **textkw,
):
    """A function to annotate a heatmap. Adapted from mpl documentation.

    Args:
        image (mpl.image.AxesImage): The AxesImage to be labeled.
        data (np.ndarray): Data used to annotate. If None, the image's data is used. Defaults to None.
        textcolors (tuple, optional): A pair of colors. The first is used for values below a threshold, the second for those above. Defaults to ("black", "white").
        threshold (float, optional): Value in data units according to which the colors from textcolors are applied. If None (the default) uses the middle of the colormap as separation. Defaults to None.
        **kwargs (dict, optional): All other arguments are forwarded to each call to `text` used to create the text labels. Defaults to {}.

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
    colorbar_label,
    x_title,
    y_title,
    data,
    x_labels,
    y_labels,
    fig,
    axes,
    image,
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
    _annotate_heatmap(image)  # type: ignore

    # Set axis labels and title
    axes.set(
        xlabel=x_title,
        ylabel=y_title,
    )

    fig.tight_layout()

    return fig, axes


def plot_sensitivity_by_time_to_outcome_heatmap(  # pylint: disable=too-many-locals
    eval_dataset: EvalDataset,
    pred_proba_thresholds: list[float],
    bins: Iterable[int] = (0, 28, 182, 365, 730, 1825),
    color_map: Optional[str] = "PuBu",
    colorbar_label: Optional[str] = "Sensitivity",
    x_title: Optional[str] = "Days to outcome",
    y_title: Optional[str] = "Positive rate",
    n_decimals_y_axis: int = 4,
    save_path: Optional[Path] = None,
) -> Union[None, Path]:
    """Plot heatmap of sensitivity by time to outcome according to different
    positive rate thresholds.

    Args:
        eval_dataset (EvalDataset): EvalDataset object.
        pred_proba_thresholds (list[float]): List of positive rate thresholds.
        bins (list, optional): Default bins for time to outcome. Defaults to [0, 1, 7, 14, 28, 182, 365, 730, 1825].
        color_map (str, optional): Colormap to use. Defaults to "PuBu".
        colorbar_label (str, optional): Colorbar label. Defaults to "Sensitivity".
        x_title (str, optional): X axis title. Defaults to "Days to outcome".
        y_title (str, optional): Y axis title. Defaults to "y_hat percentile".
        n_decimals_y_axis (int): Number of decimals to round y axis labels. Defaults to 4.
        save_path (Optional[Path], optional): Path to save the plot. Defaults to None.

    Returns:
        Union[None, Path]: None if save_path is None, else path to saved figure

    Examples:
        >>> from pathlib import Path
        >>> from psycop_model_training.utils.utils import positive_rate_to_pred_probs

        >>> repo_path = Path(__file__).parent.parent.parent.parent
        >>> path = repo_path / "tests" / "test_data" / "synth_eval_data.csv"
        >>> df = pd.read_csv(path)

        >>> positive_rate_thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

        >>> pred_proba_thresholds = positive_rate_to_pred_probs(
        >>>     pred_probs=df["pred_prob"],
        >>>     positive_rate_thresholds=positive_rate_thresholds,
        >>> )

        >>> plot_sensitivity_by_time_to_outcome(
        >>>     labels=df["label"],
        >>>     y_hat_probs=df["pred_prob"],
        >>>     pred_proba_thresholds=pred_proba_thresholds,
        >>>     outcome_timestamps=df["timestamp_t2d_diag"],
        >>>     prediction_timestamps=df["timestamp"],
        >>>     bins=[0, 28, 182, 365, 730, 1825],
        >>> )
    """
    # Construct sensitivity dataframe
    # Note that threshold_percentile IS equal to the positive rate,
    # since it is calculated on the entire dataset, not just those
    # whose true label is 1.

    func = partial(
        create_sensitivity_by_time_to_outcome_df,
        labels=eval_dataset.y,
        y_hat_probs=eval_dataset.y_hat_probs,
        outcome_timestamps=eval_dataset.outcome_timestamps,
        prediction_timestamps=eval_dataset.pred_timestamps,
        bins=bins,
    )

    df = pd.concat(
        [
            func(
                pred_proba_threshold=pred_proba_thresholds[i],
            )
            for i in range(len(pred_proba_thresholds))
        ],
        axis=0,
    )

    # Group by days_to_outcome_binned and threshold_percentile, keep only the
    # first value from each group, and reset the index
    df = (
        df.groupby(["days_to_outcome_binned", "threshold_percentile"])
        .first()
        .reset_index()
    )

    # Prepare data for plotting
    data, x_labels, y_labels = _generate_sensitivity_array(
        df,
        n_decimals_y_axis=n_decimals_y_axis,
    )

    fig, axes = plt.subplots()  # pylint: disable=invalid-name

    # Plot the heatmap
    image = axes.imshow(data, cmap=color_map)  # pylint: disable=invalid-name

    fig, axes = _format_sens_by_time_heatmap(
        colorbar_label=colorbar_label,
        x_title=x_title,
        y_title=y_title,
        data=data,
        x_labels=x_labels,
        y_labels=y_labels,
        fig=fig,
        axes=axes,
        image=image,
    )

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()
    return save_path
