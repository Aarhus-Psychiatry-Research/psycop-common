"""Loaders scz-bp outcomes"""

from typing import Literal

import pandas as pd

from psycop.common.feature_generation.loaders.raw.load_diagnoses import from_contacts
from psycop.common.feature_generation.utils import data_loaders


@data_loaders.register("manic_bipolar_outcome")
def manic_or_bipolar(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "outcome",
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


@data_loaders.register("scz_outcome")
def scz_or_sczaffective(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "outcome",
) -> pd.DataFrame:
    return from_contacts(
        icd_code=["f20", "f25"],
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )
