from pathlib import Path
from typing import Optional, Union

from psycop_model_evaluation.base_charts import (
    plot_basic_chart,
)
from psycop_model_evaluation.binary.subgroups.base import (
    create_roc_auc_by_input,
)
from psycop_model_training.training_output.dataclasses import EvalDataset


def plot_roc_auc_by_sex(
    eval_dataset: EvalDataset,
    save_path: Optional[Path] = None,
    y_limits: Optional[tuple[float, float]] = (0.5, 1.0),
    confidence_interval: Optional[float] = 0.95,
) -> Union[None, Path]:
    """Plot bar plot of performance (default AUC) by sex at time of prediction.

    Args:
        eval_dataset: EvalDataset object
        save_path: Path to save figure. Defaults to None.
        y_limits: y-axis limits. Defaults to (0.0, 1.0).
        confidence_interval: Confidence interval  width for the performance metric. Defaults to 0.95.
            by default the confidence interval is calculated by bootstrapping using
            1000 samples.

    Returns:
        Path to saved figure or None if not saved.
    """

    df = create_roc_auc_by_input(
        eval_dataset=eval_dataset,
        input_values=eval_dataset.is_female,  # type: ignore
        input_name="sex",
        bins=[0, 1],
        bin_continuous_input=False,
        confidence_interval=confidence_interval,
    )

    ci = df["ci"].tolist() if confidence_interval else None

    df.sex = df.sex.replace({1: "female", 0: "male"})

    return plot_basic_chart(
        x_values=df["sex"],
        y_values=df["metric"],
        x_title="Sex",
        y_title="AUC",
        y_limits=y_limits,
        plot_type=["bar"],
        save_path=save_path,
        confidence_interval=ci,
    )
