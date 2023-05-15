from psycop.common.model_evaluation.binary.global_performance.roc_auc import (
    plot_auc_roc,
)
from psycop.common.model_evaluation.utils import TEST_PLOT_PATH
from psycop.common.model_training.training_output.dataclasses import EvalDataset


def test_plot_roc_auc(subsampled_eval_dataset: EvalDataset):
    plot_auc_roc(
        eval_dataset=subsampled_eval_dataset,
        save_path=TEST_PLOT_PATH / "roc_auc.png",
        n_bootstraps=10,
    )
