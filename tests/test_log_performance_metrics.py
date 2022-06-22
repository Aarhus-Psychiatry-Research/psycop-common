import pytest

import pandas as pd
from src.utils import log_performance_metrics

from pathlib import Path
import wandb

@pytest.fixture(scope="function")
def eval_df():
    return pd.read_csv(Path("tests") / "test_data" / "df_synth_for_eval.csv")



def test_log_performance_metrics(eval_df):
    wandb.init(project="tests")
    log_performance_metrics(
        eval_df, 
        outcome_col_name="label",
        prediction_probabilities_col_name="pred_prob",
        id_col_name="dw_ek_borger")



