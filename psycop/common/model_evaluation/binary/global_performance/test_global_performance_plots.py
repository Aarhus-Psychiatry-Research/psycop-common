from psycop.common.model_evaluation.binary.global_performance.roc_auc import plot_auc_roc
from psycop.common.model_evaluation.utils import TEST_PLOT_PATH
from psycop.common.model_training.training_output.dataclasses import EvalDataset


def test_plot_roc_auc(subsampled_eval_dataset: EvalDataset):
    p = plot_auc_roc(eval_dataset=subsampled_eval_dataset, n_bootstraps=2)

    p.save(TEST_PLOT_PATH / "test_plot_roc_auc.png")
