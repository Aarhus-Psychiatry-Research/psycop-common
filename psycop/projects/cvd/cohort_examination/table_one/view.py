import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd
from tableone import TableOne


@dataclass
class RowSpecification:
    source_col_name: str
    readable_name: str
    categorical: bool = False
    values_to_display: Optional[Sequence[Union[int, float, str]]] = (
        None  # Which categories to display.
    )
    nonnormal: bool = False


def get_psychiatric_diagnosis_row_specs(readable_col_names: list[str]) -> list[RowSpecification]:
    # Get diagnosis columns
    pattern = re.compile(r".+f\d_disorders")
    columns = sorted(
        [c for c in readable_col_names if pattern.search(c) and "mean" in c and "730" in c]
    )

    readable_col_names = []

    for c in columns:
        icd_10_fx = re.findall(r"f\d+", c)[0]
        readable_col_name = f"{icd_10_fx.capitalize()} disorders"
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

    print(f"Adding columns {source_col_names}")

    for spec in row_specs:
        if spec.values_to_display and isinstance(spec.values_to_display[0], int):
            data[spec.source_col_name] = data[spec.source_col_name].astype(int)

    table_one = TableOne(  # type: ignore
        data=data,
        columns=source_col_names,
        categorical=[r.source_col_name for r in row_specs if r.categorical],
        groupby=groupby_col_name,
        nonnormal=[r.source_col_name for r in row_specs if r.nonnormal],
        rename={r.source_col_name: r.readable_name for r in row_specs},
        limit={r.source_col_name: r.values_to_display[0] for r in row_specs if r.values_to_display},
        order={r.source_col_name: r.values_to_display for r in row_specs if r.values_to_display},
        pval=pval,
    )

    return table_one.tableone
