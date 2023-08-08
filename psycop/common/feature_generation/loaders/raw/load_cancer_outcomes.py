"""Loaders that are specific to the cancer project."""
# pylint: disable=missing-function-docstring

from pathlib import Path
from typing import Literal

import pandas as pd

from psycop.common.feature_generation.utils import data_loaders
from psycop.common.feature_generation.loaders.raw.load_diagnoses import from_contacts


@data_loaders.register("any_cancer_outcome")
def any_cancer(
    n_rows: int | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    shak_sql_operator: str | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "outcome",
) -> pd.DataFrame:
    """Loads the outcome variable for the cancer prediction project using the common code for loading diagnoses.
    This approach gives fewer rows than the approach above (uses "FOR_kohorte_indhold_pt_journal_psyk_somatik_inkl_2021_feb2022").  
    """
    return from_contacts(
        icd_code="c",
        wildcard_icd_code=True,
        n_rows=n_rows,
        shak_location_col=shak_location_col,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
        timestamp_purpose=timestamp_purpose,
    )
