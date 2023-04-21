"""Plotting function for performance by age at time of predictio."""

from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

from psycop_model_evaluation.base_charts import (
    plot_basic_chart,
)
from psycop_model_evaluation.binary.subgroups.base import create_roc_auc_by_input
from psycop_model_training.training_output.dataclasses import EvalDataset


def plot_roc_auc_by_age(
    eval_dataset: EvalDataset,
    save_path: Optional[Path] = None,
    bins: Sequence[float] = (18, 25, 35, 50, 70),
    bin_continuous_input: Optional[bool] = True,
    y_limits: Optional[tuple[float, float]] = (0.5, 1.0),
    confidence_interval: Optional[float] = 0.95,
) -> Union[None, Path]:
    """Plot bar plot of performance (default AUC) by age at time of prediction.

    Args:
        eval_dataset: EvalDataset object
        bins: Bins to group by. Defaults to (18, 25, 35, 50, 70, 100).
        bin_continuous_input: Whether to bin input. Defaults to True.
        save_path: Path to save figure. Defaults to None.
        y_limits: y-axis limits. Defaults to (0.5, 1.0).
        confidence_interval: Confidence interval  width for the performance metric. Defaults to 0.95.
            by default the confidence interval is calculated by bootstrapping using
            1000 samples.

    Returns:
        Union[None, Path]: Path to saved figure or None if not saved.
    """

    df = create_roc_auc_by_input(
        eval_dataset=eval_dataset,
        input_values=eval_dataset.age,  # type: ignore
        input_name="age",
        bins=bins,
        bin_continuous_input=bin_continuous_input,
        confidence_interval=confidence_interval,
    )

    sort_order = sorted(df["age_binned"].unique())

    ci = df["ci"].tolist() if confidence_interval else None

    return plot_basic_chart(
        x_values=df["age_binned"],  # type: ignore
        y_values=df["metric"],  # type: ignore
        x_title="Age",
        y_title="AUC",
        sort_x=sort_order,
        y_limits=y_limits,
        plot_type=["scatter", "line"],
        bar_count_values=df["n_in_bin"],
        bar_count_y_axis_title="Number of visits",
        confidence_interval=ci,
        save_path=save_path,
    )
