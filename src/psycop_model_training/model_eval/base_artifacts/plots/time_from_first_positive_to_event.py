from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from psycop_model_training.model_eval.base_artifacts.plots.base_charts import (
    plot_basic_chart,
)
from psycop_model_training.model_eval.dataclasses import EvalDataset
from psycop_model_training.utils.utils import bin_continuous_data


def plot_time_from_first_positive_to_event(
    eval_dataset: EvalDataset,
    min_n_in_bin: Optional[int] = None,
    bins: Sequence[float] = tuple(range(0, 36, 1)),  # noqa
    fig_size: tuple[int, int] = (5, 5),
    dpi: int = 300,
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
            "y_hat_int": eval_dataset.y_hat_int,
            "y": eval_dataset.y,
            "patient_id": eval_dataset.ids,
            "pred_timestamp": eval_dataset.pred_timestamps,
            "time_from_pred_to_event": eval_dataset.outcome_timestamps
            - eval_dataset.pred_timestamps,
        },
    )

    # Get only rows where prediction is positive and outcome is positive
    df = df[(df["y_hat_int"] == 1) & (df["y"] == 1)]

    # Sort by timestamp
    df = df.sort_values("pred_timestamp")

    # Get the first positive prediction for each patient
    df = df.groupby("patient_id").first().reset_index()

    # Convert to int months
    df["time_from_pred_to_event"] = (
        df["time_from_pred_to_event"] / timedelta(days=1)  # type: ignore
    ).astype(int) / 30

    df["time_from_first_positive_to_event_binned"], _ = bin_continuous_data(
        df["time_from_pred_to_event"],
        bins=bins,
        min_n_in_bin=min_n_in_bin,
        use_min_as_label=True,
    )

    counts = df.groupby("time_from_first_positive_to_event_binned").size().reset_index()

    x_labels = counts["time_from_first_positive_to_event_binned"].to_list()
    y_values = counts[0].to_list()

    plot = plot_basic_chart(
        x_values=x_labels,
        y_values=new_var,
        x_title="Months from first positive to event",
        y_title="Count",
        plot_type="bar",
        save_path=save_path,
        sort_x=,
    )

    return plot
