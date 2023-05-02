from psycop.model_evaluation.binary.time.timedelta_plots import (
    plot_sensitivity_by_time_until_diagnosis,
)
from psycop.model_evaluation.utils import TEST_PLOT_PATH
from psycop.model_training.training_output.dataclasses import EvalDataset


def test_overlay_barplot(subsampled_eval_dataset: EvalDataset):
    plot_sensitivity_by_time_until_diagnosis(
        eval_dataset=subsampled_eval_dataset,
        y_title="Sensitivity",
        confidence_interval=0.95,
        save_path=TEST_PLOT_PATH / "test_overlay_barplot.png",
    )
