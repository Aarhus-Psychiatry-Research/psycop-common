from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import seaborn as sns

from psycop_model_training.model_eval.dataclasses import EvalDataset
from psycop_model_training.utils.utils import bin_continuous_data


def plot_time_from_first_positive_to_event(
    eval_dataset: EvalDataset,
    min_n_in_bin: Optional[int] = None,
    bins: Sequence[Union[int, float]] = tuple(range(0, 36, 1)),
    save_path: Optional[Path] = None,
) -> Union[None, Path]:
    """Plot histogram of time from first positive prediction to event.

    Args:
        eval_dataset: EvalDataset object
        min_n_in_bin (int): Minimum number of patients in each bin. If fewer, bin is dropped.
        bins (Sequence[Union[int, float]]): Bins to group by. Defaults to (5, 25, 35, 50, 70).
        save_path (Path, optional): Path to save figure. Defaults to None.
    """

    df = pd.DataFrame(
        {
            "y_hat_int": eval_dataset.y_hat_int,
            "y": eval_dataset.y,
            "patient_id": eval_dataset.ids,
            "pred_timestamp": eval_dataset.pred_timestamps,
            "time_from_first_positive_to_event": eval_dataset.outcome_timestamps
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
    df["time_from_first_positive_to_event"] = (
        df["time_from_first_positive_to_event"] / timedelta(days=1)
    ).astype(int) / 30

    df["time_from_first_positive_to_event_binned"] = bin_continuous_data(
        df["time_from_first_positive_to_event"],
        bins=bins,
        min_n_in_bin=min_n_in_bin,
        use_min_as_label=True,
    )

    # Plot a histogram of time from first positive prediction to event
    axes = sns.histplot(
        data=df,
        x="time_from_first_positive_to_event_binned",
        stat="proportion",
    )
    axes.set(
        xlabel="Months from first positive to event",
    )
    axes.invert_xaxis()
    axes.tick_params(axis="x", rotation=90)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        axes.figure.savefig(save_path, bbox_inches="tight")
        return save_path
    else:
        return None
