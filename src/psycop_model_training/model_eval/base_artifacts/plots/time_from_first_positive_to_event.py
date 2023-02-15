from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

import seaborn as sns

from psycop_model_training.model_eval.dataclasses import EvalDataset


def plot_time_from_first_positive_to_event(
    eval_dataset: EvalDataset,
    save_path: Optional[Path] = None,
) -> Union[None, Path]:
    """Plot histogram of time from first positive prediction to event."""

    df = eval_dataset.to_df()

    # Get only rows where prediction is positive and outcome is positive
    df = df[(df["y_hat_int"] == 1) & (df["y"] == 1)]

    # Get time from first positive prediction to event
    df["time_from_first_positive_to_event"] = (
        df["outcome_timestamps"] - df["pred_timestamps"]
    )

    # Convert to int days
    df["time_from_first_positive_to_event"] = (
        df["time_from_first_positive_to_event"] / timedelta(days=1)
    ).astype(int)

    # Plot a histogram of time from first positive prediction to event
    axes = sns.histplot(
        data=df,
        x="time_from_first_positive_to_event",
        stat="proportion",
    )
    axes.set(
        xlabel="Days from first positive to event",
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        axes.figure.savefig(save_path, bbox_inches="tight")
        return save_path
    else:
        return None
