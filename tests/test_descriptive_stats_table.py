"""Test that the descriptive stats table is generated correctly."""

from random import randint

import pandas as pd
from psycop_model_evaluation.descriptive_stats_table import (
    DescriptiveStatsTable,
)
from psycop_model_training.training_output.dataclasses import EvalDataset


def test_generate_descriptive_stats_table(synth_eval_dataset: EvalDataset):
    """Test that table is generated correctly."""

    col_name = "pred_f_20_max"

    additional_columns_df = pd.DataFrame(
        {
            col_name: [randint(0, 1) for _ in range(len(synth_eval_dataset.y))],
        },
    )

    table1 = DescriptiveStatsTable(
        eval_dataset=synth_eval_dataset,
        additional_columns_df=additional_columns_df,
    )

    output_table: pd.DataFrame = table1.generate_descriptive_stats_table(  # type: ignore
        output_format="df",
    )

    # Assert that there are no NaN values in the table
    assert not output_table.isnull().values.any()

    # Assert that all generic statistics are present
    assert (
        output_table["category"]
        .isin(
            [
                "(visit_level) age (mean / interval)",
                "(visit_level) visits followed by positive outcome",
                "(patient_level) patients_with_positive_outcome",
                "(patient level) time_to_first_positive_outcome",
            ],
        )
        .any()
    )

    # Assert that all additional columns are present
    assert f"(visit level) {col_name}" in output_table["category"].values
