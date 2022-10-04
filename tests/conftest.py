"""Define fixtures for tests."""
from pathlib import Path

import pandas as pd
import pytest
from utils_for_testing import add_age_gender


@pytest.fixture
def synth_data():
    """Synthetic data."""
    df = pd.read_csv(Path("tests") / "test_data" / "synth_eval_data.csv")
    df = add_age_gender(df)
    return df
