import pandas as pd
from wasabi import msg

from psycop.common.feature_generation.loaders.raw import sql_load


def load_restraint_outcome_timestamps() -> pd.DataFrame:
    """Function for loading dataframe with ids and outcome timestamps

    Returns:
        pd.DataFrame: A dataframe with ids and outcome timestamps
    """
    df = sql_load("SELECT * FROM fct.psycop_coercion_outcome_timestamps_2")

    msg.good("Finished loading data frame for coercion with prediction times and outcomes.")

    df = df.rename(
        columns={
            "datotid_start_sei": "all_restraint",
            "first_mechanical_restraint": "mechanical_restraint",
            "first_forced_medication": "chemical_restraint",
            "first_manual_restraint": "manual_restraint",
        }
    )

    return df.reset_index(drop=True)
