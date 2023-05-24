"Utility functions for feature specification"
import logging

import pandas as pd
from modules.loaders.load_forced_admissions_dfs_with_prediction_times_and_outcome import (
    LoadCoercion,
)

import wandb

log = logging.getLogger()


def add_outcome_col(flattened_df: pd.DataFrame, visit_type: str):
    """Merge outcome column into flattened dataset"""
    if visit_type == "inpatient":
        outcome_df = LoadCoercion.forced_admissions_inpatient(timestamps_only=False)
    elif visit_type == "outpatient":
        outcome_df = LoadCoercion.forced_admissions_outpatient(timestamps_only=False)
    else:
        log.info("Tried to add outcome_col, but no visit type specified")

    return pd.merge(
        flattened_df, outcome_df, how="inner", on=["dw_ek_borger", "timestamp"]
    )
