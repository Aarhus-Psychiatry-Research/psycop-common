"""Precision recall plot.""" ""
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve

from psycop_model_training.model_eval.dataclasses import EvalDataset


def plot_precision_recall(
    eval_dataset: EvalDataset,
    title: str = "",
    fig_size: Optional[tuple] = (5, 5),
    dpi: int = 160,
    save_path: Optional[Path] = None,
) -> Union[None, Path]:
    """Plot precision recall curve curve.

    Args:
        eval_dataset (EvalDataset): Evaluation dataset.
        title (str, optional): title. Defaults to "Precision-recall curve".
        fig_size (Optional[tuple], optional): figure size. Defaults to None.
        dpi (int, optional): dpi. Defaults to 160.
        save_path (Optional[Path], optional): path to save figure. Defaults to None.

    Returns:
        Union[None, Path]: None if save_path is None, else path to saved figure.
    """
    precision, recall, _ = precision_recall_curve(
        y_true=eval_dataset.y, probas_pred=eval_dataset.y_hat_probs
    )

    auprc = average_precision_score(
        y_true=eval_dataset.y, y_score=eval_dataset.y_hat_probs
    )

    legend_label = "AUPRC = "

    plt.figure(figsize=fig_size, dpi=dpi)
    plt.plot(precision, recall, label=legend_label + str(round(auprc, 3)))
    plt.legend(loc=4)

    plt.title(title)
    plt.xlabel("Precision (positive predictive value)")
    plt.ylabel("Recall (sensitivity)")

    if save_path is not None:
        if not isinstance(save_path, Path):
            save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.close()

    return save_path
