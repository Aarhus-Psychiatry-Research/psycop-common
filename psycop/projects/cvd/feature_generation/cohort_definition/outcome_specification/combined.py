import pandas as pd
import polars as pl

from psycop.common.feature_generation.loaders.raw.load_diagnoses import (
    SCORE2_CVD,
    peripheral_artery_disease,
)
from psycop.projects.cvd.feature_generation.cohort_definition.outcome_specification.procedure_codes import (
    get_cvd_procedures,
)


def get_first_cvd_indicator() -> pd.DataFrame:
    score2_diagnoses = (
        pl.from_pandas(SCORE2_CVD()).rename({"diagnosegruppestreng": "cause"}).drop("value")
    )
    pad_diagnoses = (
        pl.from_pandas(peripheral_artery_disease())
        .rename({"diagnosegruppestreng": "cause"})
        .drop("value")
    )
    procedure_codes = get_cvd_procedures().rename({"procedure_code": "cause"})

    first_cvd = (
        pl.concat([score2_diagnoses, procedure_codes, pad_diagnoses])
        .sort("timestamp")
        .groupby("dw_ek_borger")
        .first()
    )

    return first_cvd.select(["dw_ek_borger", "timestamp", "cause"]).to_pandas()


if __name__ == "__main__":
    df = get_first_cvd_indicator()
    pass
