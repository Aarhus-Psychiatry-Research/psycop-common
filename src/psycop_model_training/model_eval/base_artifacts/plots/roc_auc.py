"""AUC ROC curve."""
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
from psycop_model_training.model_eval.dataclasses import EvalDataset
from sklearn.metrics import roc_auc_score, roc_curve


def plot_auc_roc(
    eval_dataset: EvalDataset,
    title: str = "ROC-curve",
    fig_size: Optional[tuple] = (5, 5),
    dpi: int = 160,
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

    LEGEND_LABEL = "AUC = "
    
    
    plt.figure(figsize=fig_size, dpi=dpi)
    plt.plot(fpr, tpr, label=LEGEND_LABEL + str(round(auc, 3)))
    plt.legend(loc=4)
    
    plt.title(title)
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

    return save_path
