"""Loaders that are specific to the cancer project."""
# pylint: disable=missing-function-docstring

from pathlib import Path

from typing import Optional

import pandas as pd

from psycop.common.feature_generation.utils import data_loaders

# from psycop_feature_generation.loaders.raw.load_diagnoses import from_physical_visits

CANCER_DATA_DIR = Path(r"E:\shared_resources") / "cancer"


@data_loaders.register("any_cancer")
def any_cancer() -> pd.DataFrame:
    """Loads the outcome variable for the cancer prediction project.
    See `outcome_specification/outcome_specification.Rmd` for details.
    """
    df = pd.read_csv(CANCER_DATA_DIR / "cancer_cohort.csv")
    df = df.rename(columns={"datotid_start": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"]).apply(lambda x: x.replace(tzinfo=None))
    df["value"] = 1

    return df.reset_index(drop=True)

df = any_cancer()
# @data_loaders.register("any_cancer")
# def any_cancer(n_rows: Optional[int] = None) -> pd.DataFrame:
#     return from_physical_visits(
#         icd_code="C",
#         wildcard_icd_code=True,
#         n_rows=n_rows,
#     )


# @data_loaders.register("lung_cancer")
# def lung_cancer(n_rows: Optional[int] = None) -> pd.DataFrame:
#     return from_physical_visits(
#         icd_code="C34",
#         wildcard_icd_code=True,
#         n_rows=n_rows,
#     )
