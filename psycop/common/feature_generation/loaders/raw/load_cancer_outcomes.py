"""Loaders that are specific to the cancer project."""
# pylint: disable=missing-function-docstring

from typing import Literal

import pandas as pd

from psycop.common.feature_generation.loaders.raw.load_diagnoses import from_contacts
from psycop.common.feature_generation.utils import data_loaders


@data_loaders.register("any_cancer_outcome")
def any_cancer(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "outcome",
) -> pd.DataFrame:
    """Loads the outcome variable for the cancer prediction project."""
    return from_contacts(
        icd_code="c",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )


@data_loaders.register("breast_cancer_outcome")
def breast_cancer(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "outcome",
) -> pd.DataFrame:
    """Loads the outcome variable for the cancer prediction project."""
    return from_contacts(
        icd_code="c50",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )
