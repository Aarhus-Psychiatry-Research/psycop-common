import pandas as pd
import polars as pl

from psycop.common.feature_generation.loaders.raw.load_diagnoses import SCORE2_CVD
from psycop.projects.cvd.feature_generation.cohort_definition.outcome_specification.procedure_codes import (
    get_cvd_procedures,
)


def get_first_cvd_indicator() -> pd.DataFrame:
    diagnoses = pl.from_pandas(SCORE2_CVD()).rename({"diagnosegruppestreng": "cause"}).drop("value")
    procedure_codes = get_cvd_procedures().rename({"procedure_code": "cause"})

    first_cvd = (
        pl.concat([diagnoses, procedure_codes]).sort("timestamp").groupby("dw_ek_borger").first()
    )

    return first_cvd.select(["dw_ek_borger", "timestamp", "cause"]).to_pandas()


if __name__ == "__main__":
    df = get_first_cvd_indicator()
