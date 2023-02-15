from pathlib import Path
from typing import Callable, Optional, Sequence, Union

from sklearn.metrics import roc_auc_score

from psycop_model_training.model_eval.base_artifacts.plots.base_charts import (
    plot_basic_chart,
)
from psycop_model_training.model_eval.base_artifacts.plots.utils import (
    create_performance_by_input,
)
from psycop_model_training.model_eval.dataclasses import EvalDataset


def plot_time_from_first_positive_to_event(
    eval_dataset: EvalDataset,
    save_path: Optional[Path] = None,
    bins: Sequence[Union[int, float]] = (18, 25, 35, 50, 70),
    prettify_bins: Optional[bool] = True,
    metric_fn: Callable = roc_auc_score,
    y_limits: Optional[tuple[float, float]] = (0.5, 1.0),
) -> Union[None, Path]:
    """Plot histogram of time from first positive prediction to event."""

    df = eval_dataset.to_df()

    sort_order = sorted(df["age_binned"].unique())

    return plot_basic_chart(
        x_values=df["age_binned"],
        y_values=df["metric"],
        x_title="Age",
        y_title="AUC",
        sort_x=sort_order,
        y_limits=y_limits,
        plot_type=["bar"],
        save_path=save_path,
    )
