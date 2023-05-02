from pathlib import Path
from typing import Literal

import pytest
from psycop.model_evaluation.binary.time.absolute_plots import (
    plot_metric_by_absolute_time,
)
from psycop.model_evaluation.utils import TEST_PLOT_PATH
from psycop.model_training.training_output.dataclasses import EvalDataset


@pytest.mark.parametrize(
    "bin_period",
    ["Q", "Y"],
)
def test_plot_performance_by_calendar_time(
    subsampled_eval_dataset: EvalDataset,
    bin_period: Literal["M", "Q", "Y"],
):
    plot_metric_by_absolute_time(
        eval_dataset=subsampled_eval_dataset,
        bin_period=bin_period,
        confidence_interval=0.95,
        save_path=TEST_PLOT_PATH / f"test_{bin_period}.png",  # type: ignore
    )
