"""Test that the descriptive stats table is generated correctly."""


import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from psycop_ml_utils.utils_for_testing import str_to_df
from psycop_model_evaluation.descriptive_stats_table import (
    BinaryVariableSpec,
    ContinuousVariableSpec,
    ContinuousVariableToCategorical,
    DatasetSpec,
    GroupedDatasetSpec,
    VariableGroupSpec,
    _get_col_value_for_binary_row,
    _get_col_value_for_continuous_row,
    _get_col_value_for_total_row,
    _get_col_value_transform_continous_to_categorical,
    create_descriptive_stats_table,
)


@pytest.fixture()
def dataset_spec_test_split(synth_eval_df: pd.DataFrame) -> DatasetSpec:
    return DatasetSpec(name="Train", df=synth_eval_df)


@pytest.fixture()
def grouped_dataset_spec_test(synth_eval_df: pd.DataFrame) -> GroupedDatasetSpec:
    return GroupedDatasetSpec(name="Train", grouped_df=synth_eval_df)


def test_get_results_for_total_row(grouped_dataset_spec_test: GroupedDatasetSpec):
    variable_group_spec = VariableGroupSpec(
        title="Patients",
        group_column_name="dw_ek_borger",
        row_specs=["Total"],
    )

    outcome_df = _get_col_value_for_total_row(
        dataset=grouped_dataset_spec_test,
        variable_group_spec=variable_group_spec,
    )

    expected_df = str_to_df(
        """Title,Train,
Total patients,60000,
""",
    )

    assert_frame_equal(
        outcome_df,
        expected_df,
        check_dtype=False,
        check_exact=False,
        atol=10000,
    )


def test_get_results_for_binary_row(grouped_dataset_spec_test: GroupedDatasetSpec):
    row_spec = BinaryVariableSpec(
        row_title="Female",
        row_df_col_name="is_female",
        positive_class=1,
        n_decimals=None,
    )

    outcome_df = _get_col_value_for_binary_row(
        dataset=grouped_dataset_spec_test,
        row_spec=row_spec,
    )

    expected_df = str_to_df(
        """Title,Train,
Female,70%,
""",
    )

    assert_frame_equal(
        outcome_df,
        expected_df,
        check_dtype=False,
        check_exact=False,
        atol=2,
    )


def test_get_results_for_continuous_row(grouped_dataset_spec_test: GroupedDatasetSpec):
    row_spec = ContinuousVariableSpec(
        row_title="Age",
        row_df_col_name="age",
        aggregation_measure="mean",
        variance_measure="std",
        n_decimals=None,
    )

    outcome_df = _get_col_value_for_continuous_row(
        dataset=grouped_dataset_spec_test,
        row_spec=row_spec,
    )

    expected_df = str_to_df(
        """Title,Train,
Age (mean ± SD),55 ± 22,
""",
    )

    assert_frame_equal(
        outcome_df,
        expected_df,
        check_dtype=False,
        check_exact=False,
        atol=2,
    )


def test_get_col_value_for_continous_to_categorical_row(
    grouped_dataset_spec_test: GroupedDatasetSpec,
):
    row_spec = ContinuousVariableToCategorical(
        row_title="Age",
        row_df_col_name="age",
        n_decimals=None,
        bins=[18, 35, 40, 45],
        bin_decimals=None,
    )

    outcome_df = _get_col_value_transform_continous_to_categorical(
        dataset=grouped_dataset_spec_test,
        row_spec=row_spec,
    )

    expected_df = str_to_df(
        """Title,Train,
Age,,
18-35,23%,
36-40,6%,
41-45,6%,
46+,63%,
""",
    )

    assert_frame_equal(
        outcome_df,
        expected_df,
        check_dtype=False,
        check_exact=False,
        atol=2,
    )


def test_generate_descriptive_stats_table(synth_eval_df: pd.DataFrame):
    """Test descriptive stats table."""
    row_specs = [
        BinaryVariableSpec(  # The binary case
            row_title="Female",
            row_df_col_name="is_female",
            positive_class=1,
        ),
        ContinuousVariableSpec(  # The categorical case
            row_title="Age",
            row_df_col_name="age",
            aggregation_measure="mean",
            variance_measure="std",
        ),
        ContinuousVariableToCategorical(  # The continuous case
            row_title="Age",
            row_df_col_name="age",
            bins=[18, 35, 40, 45],
            bin_decimals=None,
        ),
    ]

    variable_group_specs = [
        VariableGroupSpec(
            title="Visits",
            group_column_name=None,
            add_total_row=True,
            row_specs=row_specs,  # type: ignore
        ),
        VariableGroupSpec(
            title="Patients",
            group_column_name="dw_ek_borger",
            add_total_row=True,
            row_specs=row_specs,  # type: ignore
        ),
    ]

    datasets = [
        DatasetSpec(name="Train", df=synth_eval_df),
        DatasetSpec(name="Test", df=synth_eval_df),
    ]

    descriptive_table = create_descriptive_stats_table(
        variable_group_specs=variable_group_specs,
        datasets=datasets,
    )

    print(descriptive_table)
