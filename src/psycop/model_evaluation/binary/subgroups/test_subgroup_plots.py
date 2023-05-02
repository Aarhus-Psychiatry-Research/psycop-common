from psycop.model_evaluation.binary.subgroups.age import plot_roc_auc_by_age
from psycop.model_evaluation.binary.subgroups.sex import plot_roc_auc_by_sex
from psycop.model_evaluation.utils import TEST_PLOT_PATH
from psycop.model_training.training_output.dataclasses import EvalDataset


def test_plot_performance_by_age(subsampled_eval_dataset: EvalDataset):
    plot_roc_auc_by_age(
        eval_dataset=subsampled_eval_dataset,
        save_path=TEST_PLOT_PATH / "test_performance_plot_by_age.png",
    )


def test_plot_performance_by_sex(subsampled_eval_dataset: EvalDataset):
    plot_roc_auc_by_sex(
        eval_dataset=subsampled_eval_dataset,
        save_path=TEST_PLOT_PATH / "test_performance_plot_by_sex.png",
    )
