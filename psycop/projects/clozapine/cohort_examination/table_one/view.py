import enum
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd
import polars as pl
from tableone import TableOne

from psycop.common.global_utils.cache import shared_cache
from psycop.common.model_training.utils.utils import bin_continuous_data
from psycop.projects.clozapine.cohort_examination.table_one.model import TableOneModel


class RowCategory(enum.Enum):
    demographics = enum.auto()
    outcome = enum.auto()
    diagnoses = enum.auto()
    lab_results = enum.auto()
    medications = enum.auto()


@dataclass(frozen=True)
class ColumnOverride:
    match_str: str
    categorical: bool
    category: RowCategory
    values_to_display: Sequence[int | float | str] | None
    override_name: str | None = None
    nonnormal: bool = False


@dataclass
class RowSpecification:
    source_col_name: str
    readable_name: str
    category: RowCategory
    categorical: bool = False
    values_to_display: Optional[Sequence[Union[int, float, str]]] = (
        None  # Which categories to display.
    )
    nonnormal: bool = False


def _psychiatric_fx_string_to_readable(c: str) -> str:
    """Takes a string like 'f0_disorders' and returns a readable string like 'F0 - Organic disorders'"""
    icd10_fx_to_readable = {
        "f0": "Organic disorders",
        "f1": "Substance abuse",
        "f2": "Psychotic disorders",
        "f3": "Mood disorders",
        "f4": "Neurotic and stress-related",
        "f5": "Eating and sleeping disorders",
        "f6": "Personality disorders",
        "f7": "Mental retardation",
        "f8": "Developmental disorders",
        "f9": "Child and adolescent disorders",
    }

    fx = re.findall(r"f\d+", c)[0]
    readable_col_name = f"{fx.capitalize()} - {icd10_fx_to_readable[fx]}"
    return readable_col_name


def _psychiatric_diagnosis_row_specs(
    fx_col_names: list[str], filter_regex: str = r"f\d+"
) -> list[RowSpecification]:
    # Get diagnosis columns
    columns = sorted(
        [
            c
            for c in fx_col_names
            if re.compile(filter_regex).search(c) and "bool" in c and "365" in c
        ]
    )

    fx2readable = {c: _psychiatric_fx_string_to_readable(c) for c in columns}

    return [
        RowSpecification(
            source_col_name=fx,
            readable_name=readable_col_name,
            categorical=True,
            category=RowCategory.diagnoses,
            values_to_display=[1],
        )
        for fx, readable_col_name in fx2readable.items()
    ]


@shared_cache().cache
def _to_table_one(
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
            # If only an int should be displayed, it indicates that the column is categorical
            # Fill NAs with 0, and cast to int
            data[spec.source_col_name] = data[spec.source_col_name].fillna(0).astype(int)

    table_one = TableOne(
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


def _is_prototype_column(column: str) -> bool:
    """Use only one permutation of each for the table"""
    return "bool" in column and "365" in column


def _visit_frame(model: TableOneModel, specs: Sequence[RowSpecification]) -> pd.DataFrame:
    # Order by category
    specs = sorted(specs, key=lambda x: f"{x.category.value}_{x.readable_name}")

    visits = model.frame.select(
        [r.source_col_name for r in specs if r.source_col_name not in ["age_grouped"]] + ["dataset"]
    )

    visits = visits.with_columns(
        pl.Series(
            bin_continuous_data(
                visits["pred_age_in_years"].to_pandas(), bins=[18, *list(range(19, 90, 10))]
            )[0]
        ).alias("age_grouped")  # noqa: ERA001
    )

    table = _to_table_one(row_specs=specs, data=visits.to_pandas(), groupby_col_name="dataset")

    return table


def _patient_frame(model: TableOneModel, specs: Sequence[RowSpecification]) -> pd.DataFrame:
    patient_df = model.frame.groupby("dw_ek_borger").agg(
        pl.col(model.sex_col_name).first().alias(model.sex_col_name),
        pl.col(model.outcome_col_name).max().alias(model.outcome_col_name),
        pl.col("timestamp").min().alias("first_contact_timestamp"),
        pl.col("dataset").first().alias("dataset"),
    )

    return _to_table_one(specs, data=patient_df.to_pandas(), groupby_col_name="dataset")


def clozapine_table_one(model: TableOneModel) -> pd.DataFrame:
    prototype_columns = [c for c in model.frame.columns if _is_prototype_column(c)]

    visit_frame = _visit_frame(
        model,
        [
            RowSpecification(
                source_col_name="pred_age_in_years",
                readable_name="Age",
                nonnormal=True,
                category=RowCategory.demographics,
            ),
            RowSpecification(
                source_col_name="age_grouped",
                readable_name="Age grouped",
                categorical=True,
                category=RowCategory.demographics,
            ),
            RowSpecification(
                source_col_name="pred_sex_female_fallback_0",
                readable_name="Female",
                categorical=True,
                values_to_display=[1],
                category=RowCategory.demographics,
            ),
            RowSpecification(
                source_col_name=model.outcome_col_name,
                readable_name="Incident clozapine within 365 days",
                categorical=True,
                values_to_display=[1],
                category=RowCategory.outcome,
            ),
            *_psychiatric_diagnosis_row_specs(
                fx_col_names=[c for c in prototype_columns if re.compile(r"f\d+").search(c)]
            ),
        ],
    )

    patient_frame = _patient_frame(
        model,
        [
            RowSpecification(
                source_col_name=model.sex_col_name,
                readable_name="Female",
                categorical=True,
                values_to_display=[1],
                category=RowCategory.demographics,
            ),
            RowSpecification(
                source_col_name=model.outcome_col_name,
                readable_name="Incident clozapine",
                categorical=True,
                values_to_display=[1],
                category=RowCategory.outcome,
            ),
        ],
    )

    return pd.concat([visit_frame, patient_frame], axis=0)
