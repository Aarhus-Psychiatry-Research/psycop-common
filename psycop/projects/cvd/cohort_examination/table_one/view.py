import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd
import polars as pl
from tableone import TableOne

from psycop.common.global_utils.cache import shared_cache
from psycop.common.model_evaluation.utils import bin_continuous_data
from psycop.projects.cvd.cohort_examination.label_by_outcome_type import label_by_outcome_type
from psycop.projects.cvd.cohort_examination.table_one.model import TableOneModel


@dataclass
class RowSpecification:
    source_col_name: str
    readable_name: str
    categorical: bool = False
    values_to_display: Optional[Sequence[Union[int, float, str]]] = (
        None  # Which categories to display.
    )
    nonnormal: bool = False


def _get_psychiatric_diagnosis_row_specs(readable_col_names: list[str]) -> list[RowSpecification]:
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


@shared_cache.cache
def _create_table(
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


def _visit_frame(model: TableOneModel) -> pd.DataFrame:
    specs = [
        RowSpecification(source_col_name="pred_age_in_years", readable_name="Age", nonnormal=True),
        RowSpecification(
            source_col_name="age_grouped", readable_name="Age grouped", categorical=True
        ),
        RowSpecification(
            source_col_name="pred__sex_female_fallback_0",
            readable_name="Female",
            categorical=True,
            values_to_display=[1],
        ),
        *_get_psychiatric_diagnosis_row_specs(readable_col_names=model.frame.columns),
    ]

    columns_to_keep = model.frame.select(
        [r.source_col_name for r in specs if r.source_col_name not in ["age_grouped"]] + ["dataset"]
    )

    data = columns_to_keep.to_pandas().fillna(0)
    data["age_grouped"] = pd.Series(
        bin_continuous_data(data["pred_age_in_years"], bins=[18, *list(range(19, 90, 10))])[0]
    ).astype(str)
    table = _create_table(row_specs=specs, data=data, groupby_col_name="dataset")

    return table


def _patient_frame(model: TableOneModel) -> pd.DataFrame:
    patient_row_specs = [
        RowSpecification(
            source_col_name=model.sex_col_name,
            readable_name="Female",
            categorical=True,
            values_to_display=[1],
        ),
        RowSpecification(
            source_col_name=model.outcome_col_name,
            readable_name="Incident CVD",
            categorical=True,
            values_to_display=[1],
        ),
        RowSpecification(
            source_col_name="outcome_type", readable_name="CVD by type", categorical=True
        ),
    ]

    patient_df = model.frame.groupby("dw_ek_borger").agg(
        pl.col(model.sex_col_name).first().alias(model.sex_col_name),
        pl.col(model.outcome_col_name).max().alias(model.outcome_col_name),
        pl.col("timestamp").min().alias("first_contact_timestamp"),
        pl.col("dataset").first().alias("dataset"),
        pl.col("cause").first().alias("outcome_cause"),
    )

    patient_df_labelled = label_by_outcome_type(patient_df, group_col="outcome_cause")

    return _create_table(
        patient_row_specs,
        data=patient_df_labelled.to_pandas().fillna(0),
        groupby_col_name="dataset",
    )


def table_one_view(model: TableOneModel) -> pd.DataFrame:
    visit_frame = _visit_frame(model)
    patient_frame = _patient_frame(model)

    return pd.concat([visit_frame, patient_frame], axis=0)