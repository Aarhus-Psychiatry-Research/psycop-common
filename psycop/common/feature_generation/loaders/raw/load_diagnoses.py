"""Loaders for diagnosis codes.

Is growing quite a bit, loaders may have to be split out into separate
files (e.g. psychiatric, cardiovascular, metabolic etc.) over time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from psycop.common.feature_generation.loaders.filters.cvd_filters import only_SCORE2_CVD_diagnoses
from psycop.common.feature_generation.loaders.filters.diabetes_filters import (
    keep_rows_where_diag_matches_t1d_diag,
    keep_rows_where_diag_matches_t2d_diag,
)
from psycop.common.feature_generation.loaders.raw.utils import load_from_codes

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)


def from_contacts(
    icd_code: list[str] | str,
    output_col_name: str = "value",
    code_col_name: str = "diagnosegruppestreng",
    n_rows: int | None = None,
    wildcard_icd_code: bool = False,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    keep_code_col: bool = False,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    """Load diagnoses from all hospital contacts. If icd_code is a list, will
    aggregate as one column (e.g. ["E780", "E785"] into a ypercholesterolemia
    column).

    Args:
        icd_code (str): Substring to match diagnoses for. Matches any diagnoses, whether a-diagnosis, b-diagnosis etc. # noqa: DAR102
        output_col_name (str, optional): Name of new column string. Defaults to "value".
        code_col_name (str, optional): Name of column in loaded data frame from which to extract the diagnosis codes. Defaults to "diagnosegruppestrengs".
        n_rows: Number of rows to return. Defaults to None.
        wildcard_icd_code (bool, optional): Whether to match on icd_code*. Defaults to False.
        shak_location_col (str, optional): Name of column containing shak code. Defaults to None. For diagnosis loaders, this column is "shakkode_ansvarlig". Combine with shak_code and shak_sql_operator.
        shak_code (int, optional): Shak code indicating where to keep/not keep visits from (e.g. 6600). Defaults to None.
        keep_code_col (bool, optional): Whether to keep the code column. Defaults to False.
        shak_sql_operator (str, optional): Operator indicating how to filter shak_code, e.g. "!= 6600" or "= 6600". Defaults to None.
        timestamp_purpose (Literal[str], optional): The intended use of the loader. If used as a predictor, the timestamp should be set to the contact end time, in order to avoid data leakage from future
            events. If used a an outcome, the timestamp should be set as the contact start time, in order to avoid inflation of model performance.

    Returns:
        pd.DataFrame
    """

    log.warning(
        "The DNPR3 data model replaced the DNPR2 model on 3 February 2019. Due to changes in DNPR3 granularity of diagnoses differ across the two models. If your prediction timestamps, lookbehind or lookahead span across this date, you should not use count as a resolve_multiple_fn. See the wiki (LPR2 compared to LPR3) for more information."
    )

    log.warning(
        "Diagnoses should be identified by either contact start or end time, depending on whether the diagnoses are intended as predictors or outcomes. See the wiki (OBS: Diagnosis as outcome) for more information."
    )

    allowed_timestamp_purposes = ("predictor", "outcome")
    if timestamp_purpose not in allowed_timestamp_purposes:
        raise ValueError(
            f"Invalid value for timestamp_purpose. "
            f"Allowed values are {allowed_timestamp_purposes}."
        )

    if timestamp_purpose == "predictor":
        source_timestamp_col_name = "datotid_slut"
    elif timestamp_purpose == "outcome":
        source_timestamp_col_name = "datotid_start"

    df = load_from_codes(
        codes_to_match=icd_code,
        code_col_name=code_col_name,
        source_timestamp_col_name=source_timestamp_col_name,
        view="FOR_kohorte_indhold_pt_journal_psyk_somatik_inkl_2021_feb2022",
        output_col_name=output_col_name,
        match_with_wildcard=wildcard_icd_code,
        n_rows=n_rows,
        load_diagnoses=True,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        keep_code_col=keep_code_col,
        shak_sql_operator=shak_sql_operator,
    )

    df = df.drop_duplicates(subset=["dw_ek_borger", "timestamp", output_col_name], keep="first")

    return df.reset_index(drop=True)  # type: ignore


def essential_hypertension(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="I109",
        wildcard_icd_code=False,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def hyperlipidemia(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["E780", "E785"],  # Only these two, as the others are exceedingly rare
        wildcard_icd_code=False,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def liverdisease_unspecified(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="K769",
        wildcard_icd_code=False,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def polycystic_ovarian_syndrome(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="E282",
        wildcard_icd_code=False,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def sleep_apnea(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["G473", "G4732"],
        wildcard_icd_code=False,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def sleep_problems_unspecified(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="G479",
        wildcard_icd_code=False,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def copd(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["j44"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def type_2_diabetes(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    df = from_contacts(
        icd_code=[
            "E1",
            "E16",
            "O24",
            "T383A",
            "M142",
            "G590",
            "G632",
            "H280",
            "H334",
            "H360",
            "H450",
            "N083",
        ],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
        keep_code_col=True,
    )

    df_filtered = keep_rows_where_diag_matches_t2d_diag(df=df, col_name="diagnosegruppestreng")

    return df_filtered.drop("diagnosegruppestreng", axis=1)


def ischemic_stroke(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    df = from_contacts(
        icd_code=[
            "I63",  # Cerebral infarction
            "I64",  # Stroke, not specified as hemorrhage or infarction
        ],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
        keep_code_col=True,
    )

    return df


def chronic_lung_disease(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    df = from_contacts(
        icd_code=[
            "J40",  # Bronchitis, not specified as acute or chronic
            "J41",  # Chronic bronchitis
            "J42",  # Unspecified chronic bronchitis
            "J43",  # Emphysema
            "J44",  # Other chronic obstructive pulmonary disease
        ],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
        keep_code_col=False,
    )

    return df


def acute_myocardial_infarction(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    df = from_contacts(
        icd_code=[
            "I21"  # Acute myocardial infarction
        ],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
        keep_code_col=True,
    )

    return df


def atrial_fibrillation(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    df = from_contacts(
        icd_code="I48",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )

    return df


def SCORE2_CVD(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    df = from_contacts(
        icd_code=[
            "I21",  # Acute myocardial infarction
            "I23",  # Complications after acute myocardial infarction
            "I6",  # Cerebrovascular disease
        ],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
        keep_code_col=True,
    )

    df_filtered = only_SCORE2_CVD_diagnoses(df=df, col_name="diagnosegruppestreng")

    return df_filtered


def peripheral_artery_disease(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["I702", "I739"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
        keep_code_col=True,
    )


def type_1_diabetes(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    df = from_contacts(
        icd_code=[
            "E1",
            "E16",
            "O24",
            "T383A",
            "M142",
            "G590",
            "G632",
            "H280",
            "H334",
            "H360",
            "H450",
            "N083",
        ],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
        keep_code_col=True,
    )

    df_filtered = keep_rows_where_diag_matches_t1d_diag(df=df, col_name="diagnosegruppestreng")

    return df_filtered.drop("diagnosegruppestreng", axis=1)


def angina(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    df = from_contacts(
        icd_code="I20",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )
    return df


def chronic_kidney_failure(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    df = from_contacts(
        icd_code="N18",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )
    return df


# Psychiatric diagnoses
# data loaders for all diagnoses in the f0-chapter (organic mental disorders)


def f0_disorders(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f0",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def dementia(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f00", "f01", "f02", "f03", "f04"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def delirium(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f05",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def misc_organic_mental_disorders(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f06", "f07", "f09"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


# data loaders for all diagnoses in the f1-chapter (mental and behavioural disorders due to psychoactive substance use)


def f1_disorders(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f1",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def alcohol_dependency(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f10",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def opioids_and_sedatives(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f11",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def cannabinoid_dependency(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f12",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def sedative_dependency(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f13",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def stimulant_deo(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f14", "f15"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def hallucinogen_dependency(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f16",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def tobacco_dependency(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f17",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def misc_drugs(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f18", "f19"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


# data loaders for all diagnoses in the f2-chapter (schizophrenia, schizotypal and delusional disorders)


def f2_disorders(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f2",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def schizophrenia(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f20",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def schizoaffective(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f25",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def misc_psychosis(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f21", "f22", "f23", "f24", "f28", "f29"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


# data loaders for all diagnoses in the f3-chapter (mood (affective) disorders).


def f3_disorders(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f3",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def manic_and_bipolar(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f30", "f31"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def bipolar(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f31"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def depressive_disorders(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f32", "f33", "f34", "f38"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def misc_affective_disorders(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f38", "f39"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


# data loaders for all diagnoses in the f4-chapter (neurotic, stress-related and somatoform disorders).


def f4_disorders(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f4",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def phobic_and_anxiety(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f40", "f41", "f42"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def stress_and_adjustment(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f43",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def dissociative_somatoform_and_misc(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f44", "f45", "f48"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


# data loaders for all diagnoses in the f5-chapter (behavioural syndromes associated with physiological disturbances and physical factors).


def f5_disorders(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f5",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def eating_disorders(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f50",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def sleeping_and_sexual_disorders(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f51", "f52"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def misc_f5(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f53", "f54", "f55", "f59"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


# data loaders for all diagnoses in the f6-chapter (disorders of adult personality and behaviour).


def f6_disorders(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f6",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def cluster_a(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f600", "f601"],
        wildcard_icd_code=False,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def cluster_b(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f602", "f603", "f604"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def cluster_c(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f605", "f606", "f607"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def misc_personality_disorders(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f608", "f609", "f61", "f62", "f63", "f68", "f69"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def misc_personality(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f65", "f66"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )

    # f64 sexual identity disorders is excluded


# data loaders for all diagnoses in the f7-chapter (mental retardation).


def f7_disorders(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f7",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def mild_mental_retardation(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f70",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def moderate_mental_retardation(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f71",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def severe_mental_retardation(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f72", "f73"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def misc_mental_retardation(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f78", "f79"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


# data loaders for all diagnoses in the f8-chapter (disorders of psychological development).


def f8_disorders(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f8",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def pervasive_developmental_disorders(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f84",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def misc_f8(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f80", "f81", "f82", "f83", "f88", "f89"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


# data loaders for all diagnoses in the f9-chapter (child and adolescent disorders).


def f9_disorders(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    log.warning(
        "This is a deprecated loader which also loads F99* disorders. Consider replacing with the 'f9_disorders_without_f99()' loader."
    )
    return from_contacts(
        icd_code="f9",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def f9_disorders_without_f99(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f90", "f91", "f92", "f93", "f94", "f95", "f96", "f97", "f98"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def f99_disorders(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f99",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def hyperkinetic_disorders(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f90",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def behavioural_disorders(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f91", "f92", "f93", "f94"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def tics_and_misc(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f95", "f98"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def gerd(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    """Gastroesophageal reflux disease (GERD) diagnoses."""
    return from_contacts(
        icd_code="k21",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def bipolar_a_diagnosis(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f31"],
        code_col_name="adiagnose4kar",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


def depressive_disorders_a_diagnosis(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f32", "f33", "f34", "f38"],
        code_col_name="adiagnose4kar",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )
