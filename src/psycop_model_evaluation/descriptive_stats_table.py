"""Code for generating a descriptive stats table."""
import typing as t
from typing import Optional, TypeVar, Union

import numpy as np
import pandas as pd

from psycop_model_evaluation.utils import (
    BaseModel,
    bin_continuous_data,
)


class RowSpec(BaseModel):
    row_title: str
    row_df_col_name: str
    n_decimals: Union[int, None] = 2


class BinaryRowSpec(RowSpec):
    positive_class: Union[float, str]


class CategoricalRowSpec(RowSpec):
    categories: Optional[list[str]] = None


class ContinuousRowSpec(RowSpec):
    aggregation_measure: t.Literal["mean"] = "mean"
    variance_measure: t.Literal["std"] = "std"


class ContinuousRowSpecToCategorical(RowSpec):
    bins: list[float]
    bin_decimals: Optional[int] = None


class VariableGroupSpec(BaseModel):
    title: str
    group_column_name: Optional[str]
    add_total_row: bool = True
    row_specs: Optional[list[RowSpec]] = None


class DatasetSpec(BaseModel):
    name: str
    df: pd.DataFrame


class GroupedDatasetSpec(BaseModel):
    name: str
    df: pd.DataFrame


def _create_row_df(row_title: str, col_title: str, cell_value: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Title": row_title,
            col_title: cell_value,
        },
        index=[0],
    )


def _get_col_value_for_total_row(
    dataset: GroupedDatasetSpec,
    variable_group_spec: VariableGroupSpec,
) -> pd.DataFrame:
    return _create_row_df(
        row_title=f"Total {variable_group_spec.title.lower()}",
        col_title=dataset.name,
        cell_value=dataset.df[variable_group_spec.group_column_name].nunique(),
    )


def _get_col_value_for_binary_row(
    dataset: GroupedDatasetSpec,
    row_spec: BinaryRowSpec,
) -> pd.DataFrame:
    # Get proportion with the positive class
    positive_class_prop = (
        dataset.df[row_spec.row_df_col_name] == row_spec.positive_class
    ).mean()
    prop_rounded = round(positive_class_prop * 100, row_spec.n_decimals)

    return _create_row_df(
        row_title=row_spec.row_title,
        col_title=dataset.name,
        cell_value=f"{prop_rounded}%",
    )


def _get_col_value_for_continuous_row(
    dataset: GroupedDatasetSpec,
    row_spec: ContinuousRowSpec,
) -> pd.DataFrame:
    # Aggregation
    agg_results = {
        "mean": dataset.df[row_spec.row_df_col_name].mean(),
        "median": dataset.df[row_spec.row_df_col_name].median(),
    }
    agg_result = agg_results[row_spec.aggregation_measure]
    agg_rounded = round(agg_result, row_spec.n_decimals)
    agg_cell_string = f"{agg_rounded}"

    # Variance
    variance_results = {
        "std": dataset.df[row_spec.row_df_col_name].std(),
        "iqr": dataset.df[row_spec.row_df_col_name].quantile(0.75)
        - dataset.df[row_spec.row_df_col_name].quantile(0.25),
    }
    variance_rounded = round(
        variance_results[row_spec.variance_measure],
        row_spec.n_decimals,
    )

    # Variance title
    variance_title_strings = {"std": "± SD", "iqr": "[IQR]"}
    variance_title_string = variance_title_strings[row_spec.variance_measure]

    # Variance cell
    variance_cell_strings = {
        "std": f"± {variance_rounded}",
        "iqr": f"[{agg_result - variance_rounded}, {agg_result + variance_rounded}]",
    }
    variance_cell_string = variance_cell_strings[row_spec.variance_measure]

    return _create_row_df(
        row_title=f"{row_spec.row_title} ({row_spec.aggregation_measure} {variance_title_string})",
        col_title=dataset.name,
        cell_value=f"{agg_cell_string} {variance_cell_string}",
    )


def _get_col_value_for_categorical_row():
    pass


def _get_col_value_transform_continous_to_categorical(
    dataset: GroupedDatasetSpec,
    row_spec: ContinuousRowSpecToCategorical,
) -> pd.DataFrame:
    values = bin_continuous_data(
        series=dataset.df[row_spec.row_df_col_name],
        bins=row_spec.bins,
        bin_decimals=row_spec.bin_decimals,
    )

    result_df = pd.DataFrame({"Title": values[0], "n_in_category": values[1]})

    # Get col percentage for each category within group
    grouped_df = result_df.groupby("Title").mean()
    grouped_df = grouped_df.reset_index()

    grouped_df[dataset.name] = (
        grouped_df["n_in_category"] / grouped_df["n_in_category"].sum() * 100
    )

    if row_spec.n_decimals is not None:
        grouped_df[dataset.name] = round(
            grouped_df[dataset.name],
            row_spec.n_decimals,
        )
    else:
        grouped_df[dataset.name] = grouped_df[dataset.name].astype(int)

    # Add % symbol
    grouped_df[dataset.name] = grouped_df[dataset.name].astype(str) + "%"

    # Add "Age", "" as first row
    grouped_df = pd.concat(
        [
            pd.DataFrame(
                {"Title": row_spec.row_title, dataset.name: np.nan},
                index=[0],
            ),
            grouped_df,
        ],
    )

    return grouped_df[["Title", dataset.name]]


RowSpecSubClass = TypeVar("T", bound=RowSpec)


def _process_row(
    row_spec: type[RowSpecSubClass],
    dataset: DatasetSpec,
    group_col_name: str,
) -> pd.DataFrame:
    spec_to_func = {
        BinaryRowSpec: _get_col_value_for_binary_row,
        ContinuousRowSpec: _get_col_value_for_continuous_row,
        ContinuousRowSpecToCategorical: _get_col_value_transform_continous_to_categorical,
    }

    if group_col_name is not None:
        agg_df = dataset.df.groupby(group_col_name).agg(np.mean)
        grouped_dataset_spec = GroupedDatasetSpec(name=dataset.name, df=agg_df)
    else:
        grouped_dataset_spec = GroupedDatasetSpec(
            name=dataset.name,
            df=dataset.df,
        )

    for spec in spec_to_func:
        if isinstance(row_spec, spec):
            return spec_to_func[spec](
                dataset=grouped_dataset_spec,
                row_spec=row_spec,
            )

    raise ValueError(f"Row spec type {type(row_spec)} not supported")


def _process_group(
    group: VariableGroupSpec,
    datasets: t.Sequence[DatasetSpec],
) -> pd.DataFrame:
    rows = []

    for row in group.row_specs:
        dataset_row_vals = []
        for dataset in datasets:
            dataset_row_vals.append(
                _process_row(
                    row_spec=row,
                    dataset=dataset,
                    group_col_name=group.group_column_name,
                ),
            )

        row: pd.DataFrame = dataset_row_vals[0]  # noqa: PLW2901

        for col_df in dataset_row_vals[1:]:
            row = row.merge(col_df, on="Title", how="outer")  # noqa: PLW2901

        rows.append(row)

    group_header = pd.DataFrame(
        {datasets[0].name: [f"[{group.title}]"]},
        columns=["Title"] + [dataset.name for dataset in datasets],
    )

    return pd.concat([group_header, *rows])


def create_descriptive_stats_table(
    variable_group_specs: VariableGroupSpec,
    datasets: t.Sequence[DatasetSpec],
) -> pd.DataFrame:
    groups = []

    for group in variable_group_specs:
        groups.append(_process_group(group=group, datasets=datasets))

    all_groups = pd.concat(groups)

    # Replace NaN with " "
    return all_groups.fillna(" ")
