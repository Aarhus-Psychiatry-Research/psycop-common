from psycop.model_evaluation.binary.time.timedelta_plots import (
    plot_roc_auc_by_time_from_first_visit,
    plot_sensitivity_by_time_to_event,
    plot_sensitivity_by_time_until_diagnosis,
    plot_time_from_first_positive_to_event,
)
from psycop.model_evaluation.utils import TEST_PLOT_PATH
from psycop.model_training.training_output.dataclasses import EvalDataset


def test_sensitivity_by_time_to_event(
    subsampled_eval_dataset: EvalDataset,
):
    # TODO: Another one that is currently failing, the next person who uses it should debug it
    plot_sensitivity_by_time_to_event(
        eval_dataset=subsampled_eval_dataset,
        positive_rates=[0.4, 0.6, 0.8],
        bins=list(range(0, 1460, 180)),
        n_bootstraps=10,
        y_limits=(0, 1),
        save_path=TEST_PLOT_PATH / "sensitivity_by_time_to_event.png",
    )


def test_plot_metric_until_diagnosis(subsampled_eval_dataset: EvalDataset):
    # TODO: Another one that is currently failing, the next person who uses it should debug it
    plot_sensitivity_by_time_until_diagnosis(
        eval_dataset=subsampled_eval_dataset,
        y_title="Sensitivity (recall)",
        confidence_interval=0.95,
        save_path=TEST_PLOT_PATH / "sensitivity_by_time_until_diagnosis.png",
    )


def test_plot_auc_time_from_first_visit(subsampled_eval_dataset: EvalDataset):
    plot_roc_auc_by_time_from_first_visit(
        eval_dataset=subsampled_eval_dataset,
    )


def test_plot_time_from_first_positive_to_event(subsampled_eval_dataset: EvalDataset):
    plot_time_from_first_positive_to_event(
        eval_dataset=subsampled_eval_dataset,
        bins=list(range(0, 60, 3)),
        min_n_in_bin=1,
        save_path=TEST_PLOT_PATH / "time_from_first_positive_to_event.png",
    )
