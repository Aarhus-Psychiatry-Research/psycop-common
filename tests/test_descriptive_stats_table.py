"""Test that the descriptive stats table is generated correctly."""


from pathlib import Path

import pandas as pd
import pytest
from pandas.api.types import is_datetime64_any_dtype
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
    _process_row,
    _process_top_level_group,
    create_descriptive_stats_table,
)


@pytest.fixture()
def dataset_spec_test_split(synth_eval_df: pd.DataFrame) -> DatasetSpec:
    return DatasetSpec(title="Train", df=synth_eval_df)


@pytest.fixture()
def dataset_spec(synth_eval_df: pd.DataFrame) -> DatasetSpec:
    return DatasetSpec(title="Train", df=synth_eval_df)


def test_get_results_for_total_row(dataset_spec: DatasetSpec):
    outcome_df = _get_col_value_for_total_row(
        dataset=dataset_spec,
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


def test_get_results_for_binary_row(dataset_spec: DatasetSpec):
    row_spec = BinaryVariableSpec(
        variable_title="Female",
        variable_df_col_name="is_female",
        positive_class=1,
        n_decimals=None,
    )

    outcome_df = _get_col_value_for_binary_row(
        dataset=dataset_spec,
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


def test_get_results_for_binary_row_within_group(
    synth_eval_df: pd.DataFrame,
):
    # True if the value in "timestamp_t2d_diag" is a timestamp, else false
    df = synth_eval_df

    df["any_t2d"] = df["timestamp_t2d_diag"].notna().astype(int)

    row_spec = BinaryVariableSpec(
        variable_title="Incident type 2 diabetes",
        within_group_aggregation="max",
        variable_df_col_name="any_t2d",
        positive_class=1,
    )

    expected_df = str_to_df(
        """Dataset,Title,Value
Train,Incident type 2 diabetes,9321 (14.8%),
""",
    )

    outcome_df = _process_row(
        row_spec=row_spec,
        ds=GroupedDatasetSpec(title="Train", grouped_df=df.groupby("dw_ek_borger")),
    )

    assert_frame_equal(
        outcome_df,
        expected_df,
        check_dtype=False,
        check_exact=False,
    )


test_continuos_mean_sd = (
    ContinuousVariableSpec(
        variable_title="Age",
        variable_df_col_name="age",
        aggregation_function="mean",
        variance_measure="std",
        n_decimals=None,
    ),
    str_to_df(
        """Dataset,Title,Value
Train,Age (mean ± SD),55 ± 22,
""",
    ),
)

test_median_iqr = (
    ContinuousVariableSpec(
        variable_title="Age",
        variable_df_col_name="age",
        aggregation_function="median",
        variance_measure="iqr",
        n_decimals=None,
    ),
    str_to_df(
        """Dataset,Title,Value
Train,Age (median [IQR]),56 [17; 95],
""",
    ),
)


@pytest.mark.parametrize(
    ("row_spec", "expected_df"),
    [test_continuos_mean_sd, test_median_iqr],
)
def test_get_results_for_continuous_row(
    dataset_spec: DatasetSpec,
    row_spec: ContinuousVariableSpec,
    expected_df: pd.DataFrame,
):
    outcome_df = _get_col_value_for_continuous_row(
        dataset=dataset_spec,
        row_spec=row_spec,
    )

    assert_frame_equal(
        outcome_df,
        expected_df,
        check_dtype=False,
        check_exact=False,
        atol=2,
    )


def test_get_col_value_for_continous_to_categorical_row(
    dataset_spec: DatasetSpec,
):
    row_spec = ContinuousVariableToCategorical(
        variable_title="Age",
        variable_df_col_name="age",
        n_decimals=None,
        bins=[18, 35, 40, 45],
        bin_decimals=None,
    )

    outcome_df = _get_col_value_transform_continous_to_categorical(
        dataset=dataset_spec,
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
