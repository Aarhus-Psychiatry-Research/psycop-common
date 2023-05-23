import pandas as pd
import plotnine as pn
import polars as pl
from psycop.common.model_evaluation.binary.performance_by_ppr.performance_by_ppr import (
    generate_performance_by_ppr_table,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset


def test_generate_performance_by_threshold_table(
    subsampled_eval_dataset: EvalDataset,
):
    positive_rates = [0.3, 0.2, 0.1]

    output_table = generate_performance_by_ppr_table(
        eval_dataset=subsampled_eval_dataset,
        positive_rates=positive_rates,
    )
