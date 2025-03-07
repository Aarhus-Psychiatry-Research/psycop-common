"""Loaders that are specific to the cancer project."""
# pylint: disable=missing-function-docstring

from typing import Literal

import pandas as pd

from psycop.common.feature_generation.loaders.raw.load_diagnoses import from_contacts


def lung_cancer(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "outcome",
) -> pd.DataFrame:
    """Loads the outcome variable for the lung cancer prediction project."""
    return from_contacts(
        icd_code="c34",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )
