"""Test that the descriptive stats table is generated correctly."""

from random import randint

import pandas as pd
from pandas.testing import assert_frame_equal
from psycop_ml_utils.utils_for_testing import str_to_df
from psycop_model_evaluation.descriptive_stats_table import (
    DatasetSpec,
    VariableGroupSpec,
    _get_col_value_for_total_row,
)
from psycop_model_training.training_output.dataclasses import EvalDataset


def test_get_results_for_total_row(synth_eval_df: pd.DataFrame):
    dataset_spec = DatasetSpec(name="Train", df=synth_eval_df)

    variable_group_spec = VariableGroupSpec(
        title="Patients", group_column_name="dw_ek_borger", add_total_row=True
    )

    outcome_df = _get_col_value_for_total_row(
        dataset=dataset_spec, variable_group_spec=variable_group_spec
    )

    expected_df = str_to_df(
        """Title,Train,
Total patients,60000,
"""
    )

    assert_frame_equal(
        outcome_df, expected_df, check_dtype=False, check_exact=False, atol=10000
    )


# def test_generate_descriptive_stats_table(synth_eval_dataset: EvalDataset):
#     """Test descriptive stats table."""
#     variable_group_specs = [
#         VariableGroup(
#             group_title="Patients",
#             group_column_str="dw_ek_borger",
#             add_total_row=True,
#             row_specs=[
#                 RowSpec( # The binary case
#                     row_title="Female",
#                     row_column_name="is_female",
#                 ),
#                 RowSpec( # The categorical case
#                     row_title="Female",
#                     row_column_name="is_female",
#                     categories = [0, 1],
#                 )
#                 Rowspec( # The continuous case
#                     row_title="citizen_id",
#                     row_column_name="dw_ek_borger",
#                 )
#             ],
#         )
#     ]

#     datasets = [
#         DatasetSpec(name="Train", dataset=train_df),
#         DatasetSpec(name="Test", dataset=test_df),
#     ]

#     descriptive_table = create_descriptive_stats_table(
#         variable_group_specs=variable_group_specs, datasets=datasets
#     )

#     print(descriptive_table)

#     pass
