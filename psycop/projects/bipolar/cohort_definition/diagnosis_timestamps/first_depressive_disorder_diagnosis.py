import pandas as pd
import polars as pl

from psycop.common.feature_generation.loaders.raw.load_diagnoses import (
    depressive_disorders_a_diagnosis,
)


def get_first_depressive_disorders_diagnosis() -> pd.DataFrame:
    diagnoses = pl.DataFrame(depressive_disorders_a_diagnosis())

    first_depressive_disorder = diagnoses.sort("timestamp").groupby("dw_ek_borger").first()

    return first_depressive_disorder.to_pandas()[["dw_ek_borger", "timestamp"]]


if __name__ == "__main__":
    df = get_first_depressive_disorders_diagnosis()
