from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from psycop_model_training.model_eval.base_artifacts.plots.base_charts import (
    plot_basic_chart,
)
from psycop_model_training.model_eval.dataclasses import EvalDataset
from psycop_model_training.utils.utils import bin_continuous_data


def get_top_fraction(df: pd.DataFrame, col_name: str, fraction: float):
    """
    Returns the top N percent of the data sorted by column y in a dataframe df.
    """
    # Calculate the number of rows to select
    num_rows = int(len(df) * fraction)

    # Sort the dataframe by column y and select the top N percent of rows
    sorted_df = df.sort_values(col_name, ascending=False)
    top_fraction = sorted_df.head(num_rows)

    return top_fraction


def plot_time_from_first_positive_to_event(
    eval_dataset: EvalDataset,
    min_n_in_bin: Optional[int] = None,
    bins: Sequence[float] = tuple(range(0, 36, 1)),  # noqa
    fig_size: tuple[int, int] = (5, 5),
    dpi: int = 300,
    pos_rate: float = 0.05,
    save_path: Optional[Path] = None,
) -> Union[None, Path]:
    """Plot histogram of time from first positive prediction to event.

    Args:
        eval_dataset: EvalDataset object
        min_n_in_bin (int): Minimum number of patients in each bin. If fewer, bin is dropped.
        bins (Sequence[float]): Bins to group by. Defaults to (5, 25, 35, 50, 70).
        fig_size (tuple[int, int]): Figure size. Defaults to (5,5).
        dpi (int): DPI. Defaults to 300.
        save_path (Path, optional): Path to save figure. Defaults to None.
    """

    df = pd.DataFrame(
        {
            "y_hat_probs": eval_dataset.y_hat_probs,
            "y": eval_dataset.y,
            "patient_id": eval_dataset.ids,
            "pred_timestamp": eval_dataset.pred_timestamps,
            "time_from_pred_to_event": eval_dataset.outcome_timestamps
            - eval_dataset.pred_timestamps,
        },
    )

    df_top_risk = get_top_fraction(df, "y_hat_probs", fraction=pos_rate)

    # Get only rows where prediction is positive and outcome is positive
    df_true_pos = df_top_risk[df_top_risk["y"] == 1]

    # Sort by timestamp
    df_true_pos = df_true_pos.sort_values("pred_timestamp")

    # Get the first positive prediction for each patient
    df_true_pos = df_true_pos.groupby("patient_id").first().reset_index()

    # Convert to int months
    df_true_pos["time_from_pred_to_event"] = (
        df_true_pos["time_from_pred_to_event"] / timedelta(days=1)  # type: ignore
    ).astype(int) / 30

    df_true_pos["time_from_first_positive_to_event_binned"], _ = bin_continuous_data(
        df_true_pos["time_from_pred_to_event"],
        bins=bins,
        min_n_in_bin=min_n_in_bin,
        use_min_as_label=True,
    )

    counts = (
        df_true_pos.groupby("time_from_first_positive_to_event_binned")
        .size()
        .reset_index()
    )

    x_labels = list(bins)
    y_values = counts[0].to_list()

    plot = plot_basic_chart(
        x_values=x_labels,
        y_values=y_values,
        x_title="Months from first positive to event",
        y_title="Count",
        plot_type="bar",
        save_path=save_path,
        flip_x_axis=True,
        dpi=dpi,
        fig_size=fig_size,
    )

    return plot
