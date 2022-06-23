from pathlib import Path

import pandas as pd
import pytest

# import wandb
from psycopt2d.utils import calculate_performance_metrics


@pytest.fixture(scope="function")
def synth_data():
    return pd.read_csv(Path("tests") / "test_data" / "synth_data.csv")


def test_log_performance_metrics(synth_data):
    # wandb.init(project="test")
    perf = calculate_performance_metrics(
        synth_data,
        outcome_col_name="label",
        prediction_probabilities_col_name="pred_prob",
        id_col_name="dw_ek_borger",
    )
    # stupid test - is actested in psycop-ml-utils
    # mainly used to test wandb but doesn't work GH actions
    assert type(perf) == dict
    # wandb.log(perf)
    # wandb.finish()
