import pandas as pd
from wasabi import msg

from psycop.common.feature_generation.loaders.raw import sql_load


def load_coercion_prediction_times(
    timestamps_only: bool = False,
) -> pd.DataFrame:
    """Function for loading dataframe with ids, predictions times and outcomes from SQL.

    Args:
        timestamps_only (bool, optional): Whether to only return ids and prediction times. Defaults to False.

    Returns:
        pd.DataFrame: A dataframe with ids, prediction times and potentially outcomes
    """
    df = sql_load(
        query="SELECT [adm_id],[dw_ek_borger],[timestamp_admission],[timestamp_discharge],[timestamp_outcome],[pred_adm_day_count],[timestamp],[outcome_coercion_bool_within_2_days],[outcome_coercion_type_within_2_days],[outcome_mechanical_restraint_bool_within_2_days],[outcome_chemical_restraint_bool_within_2_days],[outcome_manual_restraint_bool_within_2_days] FROM [fct].[psycop_coercion_within_2_days_feb2022]",
    )

    if timestamps_only:
        df = df[["dw_ek_borger", "timestamp"]]

    df = df.rename(columns={"outcome_timestamp": "timestamp_outcome"})

    msg.good(
        "Finished loading data frame for coercion with prediction times and outcomes.",
    )

    return df.reset_index(drop=True)
