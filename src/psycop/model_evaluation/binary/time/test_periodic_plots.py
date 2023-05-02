import pytest
from psycop.model_evaluation.binary.time.periodic_plots import (
    plot_roc_auc_by_periodic_time,
)
from psycop.model_training.training_output.dataclasses import EvalDataset


@pytest.mark.parametrize(
    "bin_period",
    ["H", "D", "M"],
)
def test_plot_performance_by_cyclic_time(
    subsampled_eval_dataset: EvalDataset,
    bin_period: str,
):
    plot_roc_auc_by_periodic_time(
        eval_dataset=subsampled_eval_dataset,
        bin_period=bin_period,
    )
