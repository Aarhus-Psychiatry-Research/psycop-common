import pandas as pd
from wasabi import msg

from psycop.common.feature_generation.loaders.raw import sql_load


def load_restraint_prediction_timestamps() -> pd.DataFrame:
    """Function for loading dataframe with ids and predictions times from SQL.
    Returns:
        pd.DataFrame: A dataframe with ids, prediction times and potentially outcomes
    """
    df = sql_load(
        "SELECT  dw_ek_borger, datotid_start as timestamp_admission, datotid_slut as timestamp_discharge, pred_adm_day_count, pred_time as timestamp FROM fct.psycop_coercion_outcome_timestamps"
    )

    msg.good("Finished loading data frame for coercion with prediction times and outcomes.")

    return df.reset_index(drop=True)
