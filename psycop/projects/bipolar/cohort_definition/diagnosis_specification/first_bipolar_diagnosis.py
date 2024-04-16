import pandas as pd
import polars as pl

from psycop.common.feature_generation.loaders.raw.load_diagnoses import bipolar_a_diagnosis


def get_first_bipolar_diagnosis() -> pd.DataFrame:
    diagnoses = pl.DataFrame(bipolar_a_diagnosis())

    first_bipolar = diagnoses.sort("timestamp").groupby("dw_ek_borger").first()

    return first_bipolar.to_pandas()[["dw_ek_borger", "timestamp"]]


if __name__ == "__main__":
    df = get_first_bipolar_diagnosis()
