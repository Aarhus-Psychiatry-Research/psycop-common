"""Test that the descriptive stats table is generated correctly."""


from pathlib import Path

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
    TotalSpec,
    VariableGroupSpec,
    _get_col_value_for_binary_row,
    _get_col_value_for_continuous_row,
    _get_col_value_for_total_row,
    _get_col_value_transform_continous_to_categorical,
    create_descriptive_stats_table,
)


@pytest.fixture()
def dataset_spec_test_split(synth_eval_df: pd.DataFrame) -> DatasetSpec:
    return DatasetSpec(title="Train", df=synth_eval_df)


@pytest.fixture()
def grouped_dataset_spec_test(synth_eval_df: pd.DataFrame) -> GroupedDatasetSpec:
    return GroupedDatasetSpec(name="Train", grouped_df=synth_eval_df)


def test_get_results_for_total_row(grouped_dataset_spec_test: GroupedDatasetSpec):
    outcome_df = _get_col_value_for_total_row(
        dataset=grouped_dataset_spec_test,
        row_spec=TotalSpec(),
    )

    expected_df = str_to_df(
        """Dataset,Title,Value
Train,Total,100000,
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
        variable_title="Female",
        variable_df_col_name="is_female",
        positive_class=1,
        n_decimals=None,
    )

    outcome_df = _get_col_value_for_binary_row(
        dataset=grouped_dataset_spec_test,
        row_spec=row_spec,
    )

    expected_df = str_to_df(
        """Dataset,Title,Value,
Train,Female,69827 (70%),
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
        variable_title="Age",
        variable_df_col_name="age",
        aggregation_measure="mean",
        variance_measure="std",
        n_decimals=None,
    )

    outcome_df = _get_col_value_for_continuous_row(
        dataset=grouped_dataset_spec_test,
        row_spec=row_spec,
    )

    expected_df = str_to_df(
        """Dataset,Title,Value
Train,Age (mean ± SD),55 ± 22,
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
        variable_title="Age",
        variable_df_col_name="age",
        n_decimals=None,
        bins=[18, 35, 40, 45],
        bin_decimals=None,
    )

    outcome_df = _get_col_value_transform_continous_to_categorical(
        dataset=grouped_dataset_spec_test,
        row_spec=row_spec,
    )

    expected_df = str_to_df(
        """Dataset,Title,Subgroup,Value
Train,Age,18-35,23058 (23%),
Train,Age,36-40,6554 (6%),
Train,Age,41-45,6388 (6%),
Train,Age,46+,62792 (63%),
""",
    )

    assert_frame_equal(
        outcome_df,
        expected_df,
        check_categorical=False,
        check_dtype=False,
        check_exact=False,
        atol=2,
    )


def test_generate_descriptive_stats_table(synth_eval_df: pd.DataFrame, tmp_path: Path):
    """Test descriptive stats table."""
    row_specs = [
        BinaryVariableSpec(  # The binary case
            variable_title="Female",
            variable_df_col_name="is_female",
            positive_class=1,
        ),
        ContinuousVariableSpec(  # The categorical case
            variable_title="Age",
            variable_df_col_name="age",
            aggregation_measure="mean",
            variance_measure="std",
        ),
        ContinuousVariableToCategorical(  # The continuous case
            variable_title="Age",
            variable_df_col_name="age",
            bins=[18, 35, 40, 45],
            bin_decimals=None,
        ),
    ]

    variable_group_specs = [
        VariableGroupSpec(
            title="Visits",
            group_column_name=None,
            add_total_row=True,
            variable_specs=row_specs,  # type: ignore
        ),
        VariableGroupSpec(
            title="Patients",
            group_column_name="dw_ek_borger",
            add_total_row=True,
            variable_specs=row_specs,  # type: ignore
        ),
    ]

    datasets = [
        DatasetSpec(title="Train", df=synth_eval_df),
        DatasetSpec(title="Test", df=synth_eval_df),
    ]

    descriptive_table = create_descriptive_stats_table(
        variable_group_specs=variable_group_specs,
        datasets=datasets,
    )

    descriptive_table.to_excel(tmp_path / "Test.xlsx")
