from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def add_age_gender(df):
    ids = pd.DataFrame({"dw_ek_borger": df["dw_ek_borger"].unique()})
    ids["age"] = np.random.randint(17, 95, len(ids))
    ids["gender"] = np.where(ids["dw_ek_borger"] > 30_000, "F", "M")

    return df.merge(ids)


@pytest.fixture
def synth_data():
    df = pd.read_csv(Path("tests") / "test_data" / "synth_data.csv")
    df = add_age_gender(df)
    return df
