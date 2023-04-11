"""Test that the descriptive stats table is generated correctly."""

from random import randint

import pandas as pd
from psycop_model_evaluation.descriptive_stats_table import (
    DescriptiveStatsTable,
)
from psycop_model_training.training_output.dataclasses import EvalDataset


def test_generate_descriptive_stats_table(synth_eval_dataset: EvalDataset):
    """Test descriptive stats table."""

    variable_group_specs = [
        VariableGroup(
            group_title="Patients",
            group_column_str="dw_ek_borger",
            add_total_row=True,
            row_specs=[
                RowSpec( # The binary case
                    row_title="Female",
                    row_column_str="is_female",
                ),
                RowSpec( # The categorical case
                    row_title="Female",
                    row_column_str="is_female",
                    categories = [0, 1],
                )
                Rowspec( # The continuous case
                    row_title="citizen_id",
                    row_column_str="dw_ek_borger",
                )
            ],
        )
    ]

    datasets = [
        DatasetSpec(dataset_name="Train", dataset=train_df),
        DatasetSpec(dataset_name="Test", dataset=test_df),
    ]

    descriptive_table = DescriptiveStatsTable(
        variable_group_specs=variable_group_specs, datasets=datasets
    )

    print(descriptive_table)

    pass
