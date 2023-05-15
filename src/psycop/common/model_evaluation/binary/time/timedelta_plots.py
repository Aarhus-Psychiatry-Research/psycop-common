import typing as t
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from pandas import Series
from psycop.common.model_evaluation.base_charts import (
    plot_basic_chart,
)
from psycop.common.model_evaluation.binary.time.timedelta_data import (
    TIMEDELTA_STRINGS,
    create_sensitivity_by_time_to_outcome_df,
    get_auroc_by_timedelta_df,
)
from psycop.common.model_evaluation.binary.utils import (
    get_top_fraction,
)
from psycop.common.model_evaluation.utils import bin_continuous_data
from psycop.common.model_training.training_output.dataclasses import EvalDataset


def plot_roc_auc_by_time_from_first_visit(
    eval_dataset: EvalDataset,
    bins: t.Sequence[float] = (0, 28, 182, 365, 730, 1825),
    bin_unit: Literal["h", "D", "M", "Q", "Y"] = "D",
    bin_continuous_input: bool = True,
    y_limits: tuple[float, float] = (0.5, 1.0),
    custom_id_to_plot_by: Optional[str] = None,
    pred_type_x_label: Optional[str] = "first visit",
    save_path: Optional[Path] = None,
) -> Union[None, Path]:
    """Plot AUC as a function of time from first visit.
    Args:
        eval_dataset (EvalDataset): EvalDataset object
        bins (list, optional): Bins to group by. Defaults to [0, 28, 182, 365, 730, 1825].
        bin_unit (Literal["h", "D", "M", "Q", "Y"], optional): Unit of time to bin by. Defaults to "D".
        bin_continuous_input (bool, optional): Whether to bin input. Defaults to True.
        y_limits (tuple[float, float], optional): Limits of y-axis. Defaults to (0.5, 1.0).
        custom_id_to_plot_by (str, optional): Custom id frome eval_dataset to plot by. If not set, it will plot by 'ids' from eval_dataset. Defaults to None.
        pred_type_x_label (str, optional): Set x label by the prediction type. Defaults to "first visit".
        save_path (Path, optional): Path to save figure. Defaults to None.
    Returns:
        Union[None, Path]: Path to saved figure or None if not saved.
    """
    if custom_id_to_plot_by:
        eval_df = pd.DataFrame(
            {
                "ids": eval_dataset.custom_columns[custom_id_to_plot_by],  # type: ignore
                "pred_timestamps": eval_dataset.pred_timestamps,
            },
        )
    else:
        eval_df = pd.DataFrame(
            {"ids": eval_dataset.ids, "pred_timestamps": eval_dataset.pred_timestamps},
        )

    first_visit_timestamps = eval_df.groupby("ids")["pred_timestamps"].transform("min")

    df = get_auroc_by_timedelta_df(
        y=eval_dataset.y,
        y_pred_proba=eval_dataset.y_hat_probs,
        time_one=first_visit_timestamps,
        time_two=eval_dataset.pred_timestamps,
        direction="t2-t1",
        bins=list(bins),
        bin_unit=bin_unit,
        bin_continuous_input=bin_continuous_input,
        drop_na_events=False,
    )

    ci = df["ci"].tolist()

    bin_unit2str = {
        "h": "Hours",
        "D": "Days",
        "M": "Months",
        "Q": "Quarters",
        "Y": "Years",
    }

    sort_order = list(range(len(df)))
    return plot_basic_chart(
        x_values=df["unit_from_event_binned"],
        y_values=df["auroc"],
        x_title=f"{bin_unit2str[bin_unit]} from {pred_type_x_label}",
        y_title="AUC",
        sort_x=sort_order,  # type: ignore
        y_limits=y_limits,
        plot_type=["line", "scatter"],
        bar_count_values=df["n_in_bin"],
        bar_count_y_axis_title="Number of visits",
        confidence_interval=ci,
        save_path=save_path,
    )


def plot_time_from_first_positive_to_event(
    eval_dataset: EvalDataset,
    min_n_in_bin: int = 0,
    bins: Sequence[float] = tuple(range(0, 36, 1)),  # noqa
    bin_unit: Literal["h", "D", "M", "Q", "Y"] = "M",
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
        bin_unit (Literal["h", "D", "M", "Q", "Y"]): Unit of time to bin by. Defaults to "M".
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
    df_true_pos["time_from_pred_to_event"] = df_true_pos[
        "time_from_pred_to_event"
    ] / np.timedelta64(
        1,
        bin_unit,
    )  # type: ignore
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

    bin_unit2str = {
        "h": "Hours",
        "D": "Days",
        "M": "Months",
        "Q": "Quarters",
        "Y": "Years",
    }

    plot = plot_basic_chart(
        x_values=x_labels,  # type: ignore
        y_values=Series(y_values),
        x_title=f"{bin_unit2str[bin_unit]} from first positive to event",
        y_title="Count",
        plot_type="bar",
        save_path=save_path,
        flip_x_axis=True,
        dpi=dpi,
        fig_size=fig_size,
    )

    return plot


def plot_sensitivity_by_time_to_event(
    eval_dataset: EvalDataset,
    positive_rates: Union[float, Iterable[float]],
    bins: Sequence[float],
    bin_unit: TIMEDELTA_STRINGS = "D",
    y_title: str = "Sensitivity (Recall)",
    save_path: Optional[Union[Path, str]] = None,
) -> Union[None, Path]:
    """Plot performance by calendar time of prediciton.
    Args:
        eval_dataset: EvalDataset object
        positive_rates: Positive rates to plot. Takes the top X% of predicted probabilities and discretises them into binary predictions.
        bins: Bins to use for time to outcome.
        bin_unit: Unit of time to bin by. Defaults to "D".
        y_title: Title of y-axis. Defaults to "AUC".
        save_path: Path to save figure. Defaults to None.
        n_bootstraps (int, optional): Number of bootstraps to use for confidence intervals. Defaults to 1000.
    Returns:
        Union[None, Path]: Path to saved figure or None if not saved.
    """
    if not isinstance(positive_rates, Iterable):
        positive_rates = [positive_rates]
    positive_rates = list(positive_rates)

    dfs = [
        create_sensitivity_by_time_to_outcome_df(
            eval_dataset=eval_dataset,
            desired_positive_rate=positive_rate,
            outcome_timestamps=eval_dataset.outcome_timestamps,
            prediction_timestamps=eval_dataset.pred_timestamps,
            bins=bins,
            bin_delta=bin_unit,
        )
        for positive_rate in positive_rates
    ]

    bin_delta_to_str = {
        "h": "Hour",
        "D": "Day",
        "W": "Week",
        "M": "Month",
        "Q": "Quarter",
        "Y": "Year",
    }

    x_title_unit = bin_delta_to_str[bin_unit]

    df = pd.concat(dfs, axis=0)

    import plotnine as pn
    from plotnine import (
        aes,
        element_text,
        geom_errorbar,
        geom_line,
        geom_point,
        ggplot,
        labs,
        scale_color_brewer,
        theme,
    )

    df["sens"] = df["sens"].astype(float)
    df["sens_lower"] = df["ci"].apply(lambda x: x["ci"][0])
    df["sens_upper"] = df["ci"].apply(lambda x: x["ci"][1])

    # Reverse order of the x axis
    df["days_to_outcome_binned"] = df["days_to_outcome_binned"].cat.reorder_categories(
        df["days_to_outcome_binned"].cat.categories[::-1],
        ordered=True,
    )

    df["actual_positive_rate"] = df["actual_positive_rate"].astype(str)

    # Set y limits

    p = (
        ggplot(
            df,
            aes(
                x="days_to_outcome_binned",
                y="sens",
                ymin="sens_lower",
                ymax="sens_upper",
                color="actual_positive_rate",
            ),
        )
        + geom_point()
        + geom_line()
        + geom_errorbar(width=0.2, size=0.5)
        + labs(x=f"{x_title_unit}s to outcome", y=y_title)
        + pn.theme_bw()
        + theme(axis_text_x=element_text(rotation=45, hjust=1))
        + scale_color_brewer(type="qual", palette=2)
        + labs(color="PPR")
        + pn.guides(color=pn.guide_legend(reverse=True))
        + pn.theme(
            panel_grid_major=pn.element_blank(),
            panel_grid_minor=pn.element_blank(),
        )
    )

    if save_path is not None:
        p.save(save_path, width=5, height=5, dpi=600)
        return Path(save_path)
    return None
