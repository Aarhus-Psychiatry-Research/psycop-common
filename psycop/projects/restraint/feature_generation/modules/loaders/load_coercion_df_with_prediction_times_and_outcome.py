import pandas as pd
from wasabi import msg

from psycop.common.feature_generation.loaders.raw import sql_load


def load_coercion_prediction_times() -> pd.DataFrame:
    """Function for loading dataframe with ids, predictions times and outcomes from SQL.

    Args:
        timestamps_only (bool, optional): Whether to only return ids and prediction times. Defaults to False.

    Returns:
        pd.DataFrame: A dataframe with ids, prediction times and potentially outcomes
    """
    df = sql_load(
        "SELECT dw_ek_borger, datotid_start as timestamp_admission, datotid_slut as timestamp_discharge, pred_adm_day_count, pred_time as timestamp FROM fct.psycop_coercion_outcome_timestamps"
    )

    df = df.rename(columns={"outcome_timestamp": "timestamp_outcome"})  # type: ignore

    msg.good("Finished loading data frame for coercion with prediction times and outcomes.")

    return df.reset_index(drop=True)
