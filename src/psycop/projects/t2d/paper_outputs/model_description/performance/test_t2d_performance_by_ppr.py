import pandas as pd
import plotnine as pn
import polars as pl
from psycop.common.model_evaluation.binary.performance_by_ppr.performance_by_ppr import (
    generate_performance_by_ppr_table,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.t2d.paper_outputs.model_description.performance.performance_by_ppr import (
    clean_up_performance_by_ppr,
)


def test_generate_performance_by_ppr_table(
    subsampled_eval_dataset: EvalDataset,
):
    positive_rates = [0.3, 0.2, 0.1]

    generated_table = generate_performance_by_ppr_table(
        eval_dataset=subsampled_eval_dataset,
        positive_rates=positive_rates,
    )

    output_table = clean_up_performance_by_ppr(table=generated_table)

    assert isinstance(output_table, pd.DataFrame)
