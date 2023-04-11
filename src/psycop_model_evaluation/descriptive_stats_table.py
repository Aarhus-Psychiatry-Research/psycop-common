"""Code for generating a descriptive stats table."""
import typing as t
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import wandb
from attr import dataclass
from psycop_model_training.training_output.dataclasses import EvalDataset

from psycop_model_evaluation.utils import (
    BaseModel,
    bin_continuous_data,
    output_table,
)


class RowSpec(BaseModel):
    row_title: str
    row_df_col_name: str
    n_decimals: Union[int, None] = 2


class BinaryRowSpec(RowSpec):
    positive_class: Union[float, str]


class CategoricalRowSpec(RowSpec):
    categories: Optional[t.List[str]] = None


class ContinuousRowSpec(RowSpec):
    aggregation_measure: t.Literal["mean"] = "mean"
    variance_measure: t.Literal["std"] = "std"


class ContinuousRowSpecToCategorical(RowSpec):
    bins: t.List[float]
    bin_decimals: Optional[int] = None


class VariableGroupSpec(BaseModel):
    title: str
    group_column_name: str
    add_total_row: bool = True
    row_specs: Optional[t.List[RowSpec]] = None


class DatasetSpec(BaseModel):
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
    dataset: DatasetSpec, variable_group_spec: VariableGroupSpec
) -> pd.DataFrame:
    return _create_row_df(
        row_title=f"Total {variable_group_spec.title.lower()}",
        col_title=dataset.name,
        cell_value=dataset.df[variable_group_spec.group_column_name].nunique(),
    )


def _get_col_value_for_binary_row(
    dataset: DatasetSpec, row_spec: BinaryRowSpec
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
    dataset: DatasetSpec, row_spec: ContinuousRowSpec
):
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
        variance_results[row_spec.variance_measure], row_spec.n_decimals
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
    dataset: DatasetSpec, row_spec: ContinuousRowSpecToCategorical
):
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
                {"Title": row_spec.row_title, dataset.name: np.nan}, index=[0]
            ),
            grouped_df,
        ]
    )

    return grouped_df[["Title", dataset.name]]


def _process_row():
    row = pd.DataFrame({"title": [row_title]})

    for dataset in datasets:
        row_type = ""  # Logic for finding the row type
        row[dataset] = ""  # Results for the given row type


def _process_group():
    for row in group.row_specs:
        _process_row(row)

    group = pd.DataFrame({"title": [group_title]})


def create_descriptive_stats_table():
    groups = []

    for group in variable_group_spec:
        groups.append(_process_group(group))

    pass
