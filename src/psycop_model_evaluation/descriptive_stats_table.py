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
    row_column_name: str
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


def _get_col_value_for_total_row(
    dataset: DatasetSpec, variable_group_spec: VariableGroupSpec
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Title": f"Total {variable_group_spec.title.lower()}",
            f"{dataset.name}": dataset.df[
                variable_group_spec.group_column_name
            ].nunique(),
        },
        index=[0],
    )

    return df


def _get_col_value_for_binary_row():
    pass


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
