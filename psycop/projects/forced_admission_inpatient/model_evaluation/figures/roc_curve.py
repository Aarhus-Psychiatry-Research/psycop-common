"""AUC ROC curve."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from psycop.common.model_training.training_output.dataclasses import EvalDataset


@dataclass
class ROCPlotSpec:
    y: pd.Series
    y_hat_probs: pd.Series
    legend_title: str


def eval_ds_to_roc_plot_spec(
    eval_dataset: EvalDataset,
    legend_title: str,
) -> ROCPlotSpec:
    """Convert EvalDataset to ROCPlotSpec."""
    return ROCPlotSpec(
        y=eval_dataset.y,  # type: ignore
        y_hat_probs=eval_dataset.y_hat_probs,  # type: ignore
        legend_title=legend_title,
    )


def plot_auc_roc(
    specs: list[ROCPlotSpec],
    title: str = "ROC-curve",
    fig_size: Optional[tuple] = (5, 5),
    dpi: int = 160,
    save_path: Optional[Path] = None,
) -> Union[None, Path]:
    """Plot AUC ROC curve."""
    plt.figure(figsize=fig_size, dpi=dpi)

    for spec in specs:
        fpr, tpr, _ = roc_curve(spec.y, spec.y_hat_probs)
        auc = roc_auc_score(spec.y, spec.y_hat_probs)
        auc_str = f"(AUC = {round(auc, 3)!s})"  # type: ignore

        plt.plot(fpr, tpr, label=f"{spec.legend_title} {auc_str}")

    plt.legend(loc=4)

    plt.title(title)
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

    return save_path
