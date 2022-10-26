"""Define fixtures for tests."""
from pathlib import Path

import pandas as pd
import pytest
from utils_for_testing import add_age_gender

from psycopt2d.evaluation_class_v2 import EvalDataset


@pytest.fixture(scope="function")
def synth_eval_dataset() -> EvalDataset:
    """Load synthetic data."""
    csv_path = Path("tests") / "test_data" / "synth_eval_data.csv"
    df = pd.read_csv(csv_path)
    df = add_age_gender(df)

    # Convert all timestamp cols to datetime
    for col in [col for col in df.columns if "timestamp" in col]:
        df[col] = pd.to_datetime(df[col])

    return EvalDataset(
        ids=df["dw_ek_borger"],
        y=df["label"],
        y_hat_probs=df["pred_prob"],
        y_hat_int=df["pred"],
        pred_timestamps=df["timestamp"],
        outcome_timestamps=df["timestamp_t2d_diag"],
        age=df["age"],
    )
