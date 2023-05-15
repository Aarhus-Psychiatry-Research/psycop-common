from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

from psycop.common.model_evaluation.binary.subgroup_data import get_auroc_by_input_df
from psycop.common.model_training.training_output.model_evaluator import EvalDataset
from psycop.projects.t2d.paper_outputs.config import EVAL_RUN, ROBUSTNESS_PATH
from psycop.projects.t2d.utils.best_runs import Run


def plot_performance_by_n_hba1c(
    eval_dataset: EvalDataset,
    save_path: Optional[Path] = None,  # noqa: ARG001
    bins: Sequence[float] = (18, 25, 35, 50, 70),
    bin_continuous_input: Optional[bool] = True,
    y_limits: Optional[tuple[float, float]] = (0.5, 1.0),  # noqa: ARG001
) -> Union[None, Path]:
    """Plot bar plot of performance (default AUC) by age at time of prediction.

    Args:
        eval_dataset: EvalDataset object
        bins (Sequence[float]): Bins to group by. Defaults to (18, 25, 35, 50, 70, 100).
        bin_continuous_input (bool, optional): Whether to bin input. Defaults to True.
        save_path (Path, optional): Path to save figure. Defaults to None.
        y_limits (tuple[float, float], optional): y-axis limits. Defaults to (0.5, 1.0).

    Returns:
        Union[None, Path]: Path to saved figure or None if not saved.
    """
    col_name = "eval_hba1c_within_9999_days_count_fallback_nan"

    df = get_auroc_by_input_df(
        eval_dataset=eval_dataset,
        input_values=eval_dataset.custom_columns[col_name],  # type: ignore
        input_name=col_name,
        bins=bins,
        bin_continuous_input=bin_continuous_input,
        confidence_interval=True,
    )

    df["ci"].tolist()

    sorted(df[f"{col_name}_binned"].unique())

    # TODO: Implement plot function


def plot_auroc_by_n_hba1c(run: Run):
    print("Plotting AUC by n HbA1c")
    eval_ds = run.get_eval_dataset(
        custom_columns=["eval_hba1c_within_9999_days_count_fallback_nan"],
    )

    plot_performance_by_n_hba1c(
        eval_dataset=eval_ds,
        bins=[0, 1, 3, 5],
        save_path=ROBUSTNESS_PATH / "auc_by_n_hba1c.png",
    )


if __name__ == "__main__":
    plot_auroc_by_n_hba1c(run=EVAL_RUN)
