import pandas as pd
from psycopmlutils.sql.loader import sql_load


def load_timestamp_for_any_diabetes():
    """Loads timestamps for the broad definition of diabetes used for wash-in.

    See R files for details.
    """
    timestamp_any_diabetes = sql_load(
        query="SELECT * FROM [fct].[psycop_t2d_first_diabetes_any]",
        format_timestamp_cols_to_datetime=False,
    )[["dw_ek_borger", "datotid_first_diabetes_any"]]

    timestamp_any_diabetes = timestamp_any_diabetes.rename(
        columns={"datotid_first_diabetes_any": "timestamp_washin"},
    )

    return timestamp_any_diabetes


def add_washin_timestamps(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add washin timestamps to dataset.

    Washin is an exclusion criterion. E.g. if the patient has any visit
    that looks like diabetes before the study starts (i.e. during
    washin), they are excluded.
    """
    timestamp_washin = load_timestamp_for_any_diabetes()

    dataset = dataset.merge(
        timestamp_washin,
        on="dw_ek_borger",
        how="left",
    )

    return dataset
