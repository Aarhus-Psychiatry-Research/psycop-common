import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd
import polars as pl
from tableone import TableOne

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load


def time_of_first_contact_to_psychiatry() -> pl.DataFrame:
    view = "FOR_kohorte_indhold_pt_journal_inkl_2021_feb2022"

    first_contact = (
        pl.from_pandas(  # type: ignore
            sql_load(  # type: ignore
                f"SELECT dw_ek_borger, datotid_start FROM [fct].[{view}]",
                format_timestamp_cols_to_datetime=True,  # type: ignore
                n_rows=None,  # type: ignore
            )
        )
        .groupby("dw_ek_borger")
        .agg(pl.col("datotid_start").min().alias("first_contact"))
    )
    return first_contact


@dataclass
class RowSpecification:
    source_col_name: str
    readable_name: str
    categorical: bool = False
    values_to_display: Optional[Sequence[Union[int, float, str]]] = None
    nonnormal: bool = False


def get_psychiatric_diagnosis_row_specs(readable_col_names: list[str]) -> list[RowSpecification]:
    # Get diagnosis columns
    pattern = re.compile(r"pred_f\d_disorders")
    columns = sorted(
        [c for c in readable_col_names if pattern.search(c) and "boolean" in c and "365" in c]
    )

    readable_col_names = []

    for c in columns:
        icd_10_fx = re.findall(r"f\d+", c)[0]
        readable_col_name = f"{icd_10_fx.capitalize()} disorders prior 1 year"
        readable_col_names.append(readable_col_name)

    specs = []
    for i, _ in enumerate(readable_col_names):
        specs.append(
            RowSpecification(
                source_col_name=columns[i],
                readable_name=readable_col_names[i],
                categorical=True,
                nonnormal=False,
                values_to_display=[1],
            )
        )

    return specs


def create_table(
    row_specs: Sequence[RowSpecification],
    data: pd.DataFrame,
    groupby_col_name: str,
    pval: bool = False,
) -> pd.DataFrame:
    """Unpacks a sequence of row specs into the format used by TableOne"""
    source_col_names = [r.source_col_name for r in row_specs]
    categorical_col_names = [r.source_col_name for r in row_specs if r.categorical]
    nonnormal_col_names = [r.source_col_name for r in row_specs if r.nonnormal]
    readable_col_names = {r.source_col_name: r.readable_name for r in row_specs}

    limit_columns = {
        r.source_col_name: r.values_to_display[0] for r in row_specs if r.values_to_display
    }
    order_columns = {
        r.source_col_name: r.values_to_display for r in row_specs if r.values_to_display
    }

    print(f"Adding columns {source_col_names}")

    for spec in row_specs:
        if spec.values_to_display and isinstance(spec.values_to_display[0], int):
            data[spec.source_col_name] = data[spec.source_col_name].astype(int)

    table_one = TableOne(  # type: ignore
        data=data,
        columns=source_col_names,
        categorical=categorical_col_names,
        groupby=groupby_col_name,
        nonnormal=nonnormal_col_names,
        rename=readable_col_names,
        limit=limit_columns,
        order=order_columns,
        pval=pval,
    )

    return table_one.tableone
