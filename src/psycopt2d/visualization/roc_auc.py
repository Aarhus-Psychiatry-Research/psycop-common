"""AUC ROC curve."""
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from psycopt2d.evaluation_dataclasses import EvalDataset


def plot_auc_roc(
    eval_dataset: EvalDataset,
    fig_size: Optional[tuple] = (10, 10),
    save_path: Optional[Path] = None,
) -> Union[None, Path]:
    """Plot AUC ROC curve.

    Args:
        eval_dataset (EvalDataset): Evaluation dataset.
        fig_size (Optional[tuple], optional): figure size. Defaults to None.
        save_path (Optional[Path], optional): path to save figure. Defaults to None.

    Returns:
        Union[None, Path]: None if save_path is None, else path to saved figure.
    """
    fpr, tpr, _ = roc_curve(eval_dataset.y, eval_dataset.y_hat_probs)
    auc = roc_auc_score(eval_dataset.y, eval_dataset.y_hat_probs)

    plt.figure(figsize=fig_size)
    plt.plot(fpr, tpr, label="AUC score = " + str(auc))
    plt.title("AUC ROC Curve")
    plt.legend(loc=4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

    return save_path
