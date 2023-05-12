from pathlib import Path

from psycop.common.model_evaluation.binary.global_performance.precision_recall import (
    plot_precision_recall,
)
from psycop.common.model_evaluation.binary.global_performance.roc_auc import (
    plot_auc_roc,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset


def test_plot_roc_auc(subsampled_eval_dataset: EvalDataset, test_plot_path: Path):
    plot_auc_roc(
        eval_dataset=subsampled_eval_dataset,
        save_path=test_plot_path / "roc_auc.png",
        n_bootstraps=10,
    )


def test_plot_precision_recall(
    subsampled_eval_dataset: EvalDataset,
    test_plot_path: Path,
):
    plot_precision_recall(
        eval_dataset=subsampled_eval_dataset,
        save_path=Path(
            test_plot_path / "precision_recall.png",
        ),
    )
