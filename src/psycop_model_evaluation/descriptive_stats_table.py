"""Code for generating a descriptive stats table."""
import typing as t
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from psycop_model_evaluation.utils import (
    BaseModel,
    bin_continuous_data,
)


class VariableSpec(BaseModel):
    _name: str = "Base"
    variable_title: str  # Title for the created row in the table
    variable_df_col_name: str  # Source column name in the dataset df.
    n_decimals: Union[int, None] = 1  # Number of decimals to round the results to
    within_group_aggregation: Optional[t.Literal["mean", "median", "max", "min"]] = None


class TotalSpec(VariableSpec):
    _name: str = "Total"
    variable_title: str = "Total"
    variable_df_col_name: str = "Total"
    n_decimals: Union[int, None] = None


class BinaryVariableSpec(VariableSpec):
    _name: str = "Binary"
    positive_class: Union[
        float,
        str,
    ]  # Value of the class to generate results for (e.g. 1 for a binary variable)


class CategoricalVariableSpec(VariableSpec):
    _name: str = "Categorical"
    categories: Optional[list[str]] = None  # List of categories to include in the table


class ContinuousVariableSpec(VariableSpec):
    _name: str = "Continuous"
    aggregation_function: t.Literal["mean", "median"] = "mean"
    variance_measure: t.Literal["std", "iqr"] = "std"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.set_variable_title()

    def set_variable_title(self):
        variance_title_strings = {"std": "± SD", "iqr": "[IQR]"}
        variance_title_string = variance_title_strings[self.variance_measure]

        self.Config.allow_mutation = True
        self.variable_title = f"{self.variable_title} ({self.aggregation_function} {variance_title_string})"
        self.Config.allow_mutation = False


class ContinuousVariableToCategorical(VariableSpec):
    _name: str = "ContinuousToCategorical"
    bins: list[float]  # List of bin edges
    bin_decimals: Optional[int] = None  # Number of decimals to round the bin edges to


class VariableGroupSpec(BaseModel):
    title: str  # Title to add to the table
    group_column_name: Optional[str]  # Column name to group by
    add_total_row: bool = True  # Whether to add a total row, e.g. "100_000 patients"
    variable_specs: list[VariableSpec]  # List of row specs to include in the table


class DatasetSpec(BaseModel):
    title: str  # Name of the dataset, used as a column name in the table
    df: pd.DataFrame


class GroupedDatasetSpec(BaseModel):
    title: str  # Name of the dataset, used as a column name in the table
    grouped_df: pd.core.groupby.DataFrameGroupBy  # type: ignore


def _create_row_df(
    dataset_title: str,
    value_title: str,
    cell_value: Union[float, str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Dataset": dataset_title,
            "Title": value_title,
            "Value": cell_value,
        },
        index=[0],
    )


def _get_col_value_for_total_row(
    dataset: DatasetSpec,
    row_spec: TotalSpec,  # noqa: ARG001, reason: row_spec is not used, but required to maintain consistent function signatures for application from dict
) -> pd.DataFrame:
    # Get number of rows in grouped df
    n_rows_by_group = dataset.df.shape[0]

    return _create_row_df(
        value_title="Total",
        dataset_title=dataset.title,
        cell_value=n_rows_by_group,
    )


def _get_col_value_for_binary_row(
    dataset: DatasetSpec,
    row_spec: BinaryVariableSpec,
) -> pd.DataFrame:
    # Get proportion with the positive class
    positive_class_prop = (
        dataset.df[row_spec.variable_df_col_name] == row_spec.positive_class
    ).mean()
    prop_rounded = round(positive_class_prop * 100, row_spec.n_decimals)
    n_rows = (
        dataset.df[row_spec.variable_df_col_name] == row_spec.positive_class
    ).sum()

    return _create_row_df(
        value_title=row_spec.variable_title,
        dataset_title=dataset.title,
        cell_value=f"{n_rows} ({prop_rounded}%)",
    )


def _get_col_value_for_continuous_row(
    dataset: DatasetSpec,
    row_spec: ContinuousVariableSpec,
) -> pd.DataFrame:
    # Aggregation
    agg_results = {
        "mean": dataset.df[row_spec.variable_df_col_name].mean(),
        "median": dataset.df[row_spec.variable_df_col_name].median(),
    }
    agg_result = agg_results[row_spec.aggregation_function]
    agg_rounded = round(agg_result, row_spec.n_decimals)
    agg_cell_string = f"{agg_rounded}"

    # Variance
    variance_results = {
        "std": dataset.df[row_spec.variable_df_col_name].std(),
        "iqr": dataset.df[row_spec.variable_df_col_name].quantile(0.75)
        - dataset.df[row_spec.variable_df_col_name].quantile(0.25),
    }
    variance_rounded = round(
        variance_results[row_spec.variance_measure],
        row_spec.n_decimals,
    )

    # Variance cell
    variance_cell_strings = {
        "std": f"± {variance_rounded}",
        "iqr": f"[{round_support_no_decimals(agg_result - variance_rounded, n_decimals=row_spec.n_decimals)}; {round_support_no_decimals(agg_result + variance_rounded, n_decimals=row_spec.n_decimals)}]",
    }
    variance_cell_string = variance_cell_strings[row_spec.variance_measure]

    return _create_row_df(
        value_title=row_spec.variable_title,
        dataset_title=dataset.title,
        cell_value=f"{agg_cell_string} {variance_cell_string}",
    )


def round_support_no_decimals(x: float, n_decimals: Optional[int]) -> Union[int, float]:
    if n_decimals is None:
        return int(x)
    return round(x, n_decimals)


def _get_col_value_for_categorical_row():
    # Refactor from _get_col_value_transform_continous_to_categorical, since it basically handles
    # a categorical column
    pass


def _get_col_value_transform_continous_to_categorical(
    dataset: DatasetSpec,
    row_spec: ContinuousVariableToCategorical,
) -> pd.DataFrame:
    values = bin_continuous_data(
        series=dataset.df[row_spec.variable_df_col_name],
        bins=row_spec.bins,
        bin_decimals=row_spec.bin_decimals,
    )

    result_df = pd.DataFrame({"Subgroup": values[0], "n_in_category": values[1]})

    # Get col percentage for each category within group
    binned_df = result_df.groupby("Subgroup").mean()
    binned_df["Dataset"] = dataset.title
    binned_df["Title"] = row_spec.variable_title

    binned_df = binned_df.reset_index()

    percent_in_category = (
        binned_df["n_in_category"] / binned_df["n_in_category"].sum() * 100
    )

    percent_in_category_str = (
        round_series(percent_in_category, decimals=row_spec.n_decimals).astype(str)
        + "%"
    )
    n_in_category_str = round_series(binned_df["n_in_category"], decimals=None).astype(
        str,
    )

    # Convert to a nice string
    binned_df["Value"] = n_in_category_str + " (" + percent_in_category_str + ")"

    return binned_df[["Dataset", "Title", "Subgroup", "Value"]]


def round_series(series: pd.Series, decimals: Optional[int]) -> pd.Series:
    if decimals is None:
        return series.astype(int)
    return series.round(decimals=decimals)


def _process_row(
    row_spec: VariableSpec,
    ds: Union[GroupedDatasetSpec, DatasetSpec],
) -> pd.DataFrame:
    # Handle aggregation of grouped dfs
    if isinstance(ds, GroupedDatasetSpec):
        agg_str2fn = {
            "mean": np.mean,
            "max": np.max,
            "min": np.min,
            "median": np.median,
        }

        ds = DatasetSpec(
            title=ds.title,
            df=ds.grouped_df.agg(agg_str2fn[row_spec.within_group_aggregation]),  # type: ignore
        )

    spec_to_func = {
        "Binary": _get_col_value_for_binary_row,
        "Continuous": _get_col_value_for_continuous_row,
        "ContinuousToCategorical": _get_col_value_transform_continous_to_categorical,
        "Total": _get_col_value_for_total_row,
    }

    return spec_to_func[row_spec._name](  # type: ignore
        dataset=ds,
        row_spec=row_spec,
    )


def _create_title_row(
    group_spec: VariableGroupSpec,
    dataset: DatasetSpec,
) -> pd.DataFrame:
    return _create_row_df(
        value_title="Observation unit",
        dataset_title=dataset.title,
        cell_value=f"[{group_spec.title}]",
    )


def _process_top_level_group(
    group_spec: VariableGroupSpec,
    datasets: t.Sequence[DatasetSpec],
) -> pd.DataFrame:
    rows = [_create_title_row(group_spec, dataset) for dataset in datasets]

    if group_spec.add_total_row:
        # Add total to the front of the row specs
        group_spec.variable_specs.insert(
            0,
            TotalSpec(),
        )

    group_col_name = group_spec.group_column_name

    for ds in datasets:
        # Handle grouping if relevant
        if group_col_name is not None:
            df = ds.df.groupby(group_col_name)
            ds_spec = GroupedDatasetSpec(title=ds.title, grouped_df=df)
        else:
            ds_spec = DatasetSpec(title=ds.title, df=ds.df)

        for row_spec in group_spec.variable_specs:
            rows.append(
                _process_row(
                    row_spec=row_spec,
                    ds=ds_spec,
                ),
            )

    # Pivot into the right shape
    dataset_rows = pd.concat(rows).reset_index(drop=True)

    if "Subgroup" not in dataset_rows.columns:
        dataset_rows["Subgroup"] = ""

    table = dataset_rows.pivot(
        index=["Title", "Subgroup"],
        columns="Dataset",
        values="Value",
    )

    # Re-order to match spec order in input
    title_order = [
        variable_spec.variable_title for variable_spec in group_spec.variable_specs
    ]
    title_order.insert(0, "Observation unit")
    table = table.reindex(title_order, level=0)

    return table


def create_descriptive_stats_table(
    variable_group_specs: t.Sequence[VariableGroupSpec],
    datasets: t.Sequence[DatasetSpec],
) -> pd.DataFrame:
    groups = []

    for group_spec in variable_group_specs:
        groups.append(
            _process_top_level_group(group_spec=group_spec, datasets=datasets),
        )

    all_groups = pd.concat(groups)

    # Fill in missing values (np.Nan and all other values) with an empty string
    with_index = all_groups.reset_index()
    with_index["Subgroup"] = with_index["Subgroup"].astype(str).replace("nan", "")

    # If "Title" is repeating, only keep the first occurence
    with_index["Title"] = with_index["Title"].where(
        with_index["Title"].shift() != with_index["Title"],
        "",
    )
    with_index["Title"] = with_index["Title"].where(
        with_index["Title"] != "Observation unit",
        "",
    )

    return with_index
