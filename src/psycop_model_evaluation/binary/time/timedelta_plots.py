from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd
from pandas import Series
from psycop_model_evaluation.base_charts import (
    plot_basic_chart,
)
from psycop_model_evaluation.binary.time.timedelta_data import (
    create_performance_by_timedelta,
)
from psycop_model_evaluation.binary.utils import (
    get_top_fraction,
)
from psycop_model_evaluation.utils import bin_continuous_data
from psycop_model_training.model_eval.dataclasses import EvalDataset
from sklearn.metrics import recall_score, roc_auc_score


def plot_roc_auc_by_time_from_first_visit(
    eval_dataset: EvalDataset,
    bins: tuple = (0, 28, 182, 365, 730, 1825),
    bin_unit: Literal["H", "D", "M", "Q", "Y"] = "D",
    bin_continuous_input: bool = True,
    y_limits: tuple[float, float] = (0.5, 1.0),
    save_path: Optional[Path] = None,
) -> Union[None, Path]:
    """Plot AUC as a function of time from first visit.
    Args:
        eval_dataset (EvalDataset): EvalDataset object
        bins (list, optional): Bins to group by. Defaults to [0, 28, 182, 365, 730, 1825].
        bin_unit (Literal["H", "D", "M", "Q", "Y"], optional): Unit of time to bin by. Defaults to "D".
        bin_continuous_input (bool, optional): Whether to bin input. Defaults to True.
        y_limits (tuple[float, float], optional): Limits of y-axis. Defaults to (0.5, 1.0).
        save_path (Path, optional): Path to save figure. Defaults to None.
    Returns:
        Union[None, Path]: Path to saved figure or None if not saved.
    """
    eval_df = pd.DataFrame(
        {"ids": eval_dataset.ids, "pred_timestamps": eval_dataset.pred_timestamps},
    )

    first_visit_timestamps = eval_df.groupby("ids")["pred_timestamps"].transform("min")

    df = create_performance_by_timedelta(
        y=eval_dataset.y,
        y_to_fn=eval_dataset.y_hat_probs,
        metric_fn=roc_auc_score,
        time_one=first_visit_timestamps,
        time_two=eval_dataset.pred_timestamps,
        direction="t2-t1",
        bins=list(bins),
        bin_unit=bin_unit,
        bin_continuous_input=bin_continuous_input,
        drop_na_events=False,
    )

    bin_unit2str = {
        "H": "Hours",
        "D": "Days",
        "M": "Months",
        "Q": "Quarters",
        "Y": "Years",
    }

    sort_order = list(range(len(df)))
    return plot_basic_chart(
        x_values=df["unit_from_event_binned"],
        y_values=df["metric"],
        x_title=f"{bin_unit2str[bin_unit]} from first visit",
        y_title="AUC",
        sort_x=sort_order,  # type: ignore
        y_limits=y_limits,
        plot_type=["line", "scatter"],
        bar_count_values=df["n_in_bin"],
        bar_count_y_axis_title="Number of visits",
        save_path=save_path,
    )


def plot_sensitivity_by_time_until_diagnosis(
    eval_dataset: EvalDataset,
    bins: Sequence[int] = (
        -1825,
        -730,
        -365,
        -182,
        -28,
        -0,
    ),
    bin_unit: Literal["H", "D", "M", "Q", "Y"] = "D",
    bin_continuous_input: bool = True,
    positive_rate: float = 0.5,
    y_title: str = "Sensitivity (recall)",
    y_limits: Optional[tuple[float, float]] = None,
    save_path: Optional[Path] = None,
) -> Union[None, Path]:
    """Plots performance of a specified performance metric in bins of time
    until diagnosis. Rows with no date of diagnosis (i.e. no outcome) are
    removed.
    Args:
        eval_dataset (EvalDataset): EvalDataset object
        bins (list, optional): Bins to group by. Negative values indicate days after
        bin_unit (Literal["H", "D", "M", "Q", "Y"], optional): Unit of time to bin by. Defaults to "D".
        diagnosis. Defaults to (-1825, -730, -365, -182, -28, -14, -7, -1, 0)
        bin_continuous_input (bool, optional): Whether to bin input. Defaults to True.
        positive_rate (float, optional): Takes the top positive_rate% of predicted probabilities and turns them into 1, the rest 0.
        y_title (str): Title for y-axis (metric name)
        y_limits (tuple[float, float], optional): Limits of y-axis. Defaults to None.
        save_path (Path, optional): Path to save figure. Defaults to None.
    Returns:
        Union[None, Path]: Path to saved figure if save_path is specified, else None
    """
    df = create_performance_by_timedelta(
        y=eval_dataset.y,
        y_to_fn=eval_dataset.get_predictions_for_positive_rate(
            desired_positive_rate=positive_rate,
        )[0],
        metric_fn=recall_score,
        time_one=eval_dataset.outcome_timestamps,
        time_two=eval_dataset.pred_timestamps,
        direction="t1-t2",
        bins=bins,
        bin_unit=bin_unit,
        bin_continuous_input=bin_continuous_input,
        min_n_in_bin=5,
        drop_na_events=True,
    )
    sort_order = list(range(len(df)))

    bin_unit2str = {
        "H": "Hours",
        "D": "Days",
        "M": "Months",
        "Q": "Quarters",
        "Y": "Years",
    }

    return plot_basic_chart(
        x_values=df["unit_from_event_binned"],
        y_values=df["metric"],
        x_title=f"{bin_unit2str[bin_unit]} to diagnosis",
        y_title=y_title,
        sort_x=sort_order,
        bar_count_values=df["n_in_bin"],
        y_limits=y_limits,
        plot_type=["scatter", "line"],
        save_path=save_path,
    )


def plot_time_from_first_positive_to_event(
    eval_dataset: EvalDataset,
    min_n_in_bin: int = 0,
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
        pos_rate (float): Desired positive rate for computing which threshold above which a prediction is marked as "positive". Defaults to 0.05.
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

    x_labels = list(counts["time_from_first_positive_to_event_binned"])
    y_values = counts[0].to_list()

    plot = plot_basic_chart(
        x_values=x_labels,  # type: ignore
        y_values=Series(y_values),
        x_title="Months from first positive to event",
        y_title="Count",
        plot_type="bar",
        save_path=save_path,
        flip_x_axis=True,
        dpi=dpi,
        fig_size=fig_size,
    )

    return plot
