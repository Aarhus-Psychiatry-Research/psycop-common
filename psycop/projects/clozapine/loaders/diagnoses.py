"""Loaders for diagnosis codes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import pandas as pd

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
    keep_code_col: bool = False,
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
        keep_code_col (bool, optional): Whether to keep the code column. Defaults to False.
        timestamp_purpose (Literal[str], optional): The intended use of the loader. If used as a predictor, the timestamp should be set to the contact end time, in order to avoid data leakage from future
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
        view="Clozapin_kohorte_indhold",
        output_col_name=output_col_name,
        match_with_wildcard=wildcard_icd_code,
        n_rows=n_rows,
        load_diagnoses=True,
        keep_code_col=keep_code_col,
    )

    df = df.drop_duplicates(subset=["dw_ek_borger", "timestamp", output_col_name], keep="first")

    return df.reset_index(drop=True)  # type: ignore


def schizophrenia(
    n_rows: int | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f20", wildcard_icd_code=True, n_rows=n_rows, timestamp_purpose=timestamp_purpose
    )


def schizoaffective(
    n_rows: int | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f25", wildcard_icd_code=True, n_rows=n_rows, timestamp_purpose=timestamp_purpose
    )


def f0_disorders(
    n_rows: int | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f0", wildcard_icd_code=True, n_rows=n_rows, timestamp_purpose=timestamp_purpose
    )


def f1_disorders(
    n_rows: int | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f1", wildcard_icd_code=True, n_rows=n_rows, timestamp_purpose=timestamp_purpose
    )


def f2_disorders(
    n_rows: int | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f2", wildcard_icd_code=True, n_rows=n_rows, timestamp_purpose=timestamp_purpose
    )


def f3_disorders(
    n_rows: int | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f3", wildcard_icd_code=True, n_rows=n_rows, timestamp_purpose=timestamp_purpose
    )


def manic_and_bipolar(
    n_rows: int | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f30", "f31"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        timestamp_purpose=timestamp_purpose,
    )


def f4_disorders(
    n_rows: int | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f4", wildcard_icd_code=True, n_rows=n_rows, timestamp_purpose=timestamp_purpose
    )


def f5_disorders(
    n_rows: int | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f5", wildcard_icd_code=True, n_rows=n_rows, timestamp_purpose=timestamp_purpose
    )


def f6_disorders(
    n_rows: int | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f6", wildcard_icd_code=True, n_rows=n_rows, timestamp_purpose=timestamp_purpose
    )


def cluster_b(
    n_rows: int | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f602", "f603", "f604"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        timestamp_purpose=timestamp_purpose,
    )


def f7_disorders(
    n_rows: int | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f7", wildcard_icd_code=True, n_rows=n_rows, timestamp_purpose=timestamp_purpose
    )


def f8_disorders(
    n_rows: int | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f8", wildcard_icd_code=True, n_rows=n_rows, timestamp_purpose=timestamp_purpose
    )


def f9_disorders_without_f99(
    n_rows: int | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f90", "f91", "f92", "f93", "f94", "f95", "f96", "f97", "f98"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        timestamp_purpose=timestamp_purpose,
    )


def f99_disorders(
    n_rows: int | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    return from_contacts(
        icd_code="f99", wildcard_icd_code=True, n_rows=n_rows, timestamp_purpose=timestamp_purpose
    )
