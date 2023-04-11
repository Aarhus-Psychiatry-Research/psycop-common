"""Code for generating a descriptive stats table."""
import typing as t
import warnings
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import wandb
from attr import dataclass
from psycop_model_training.training_output.dataclasses import EvalDataset

from psycop_model_evaluation.utils import (
    bin_continuous_data,
    output_table,
)


@dataclass
class RowSpec:
    row_title: str
    row_df_col_name: str


@dataclass
class BinaryRowSpec(RowSpec):
    positive_class: Union[str, float]
    n_decimals: int = 2


@dataclass
class CategoricalRowSpec(RowSpec):
    categories: Optional[t.List[str]] = None


@dataclass
class VariableGroupSpec:
    title: str
    group_column_name: str
    add_total_row: bool = True
    row_specs: Optional[t.List[RowSpec]] = None


@dataclass
class DatasetSpec:
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
    percent_value = round(positive_class_prop * 100, row_spec.n_decimals)

    if row_spec.n_decimals == 0:
        percent_value = int(percent_value)

    return _create_row_df(
        row_title=row_spec.row_title,
        col_title=dataset.name,
        cell_value=f"{percent_value}%",
    )


def _get_col_value_for_continuous_row():
    pass


def _get_col_value_for_categorical_row():
    pass


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
