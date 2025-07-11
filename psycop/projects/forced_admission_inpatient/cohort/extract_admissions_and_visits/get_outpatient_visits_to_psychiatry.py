"""
Script for obtaining and writing all admission  and discharge timestamps
for all admissions from 2012-2022. Handles LPR2 to LPR3 transition, duplicates
and short term readmissions
"""

from datetime import timedelta

import pandas as pd

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load


def get_outpatient_visits_to_psychiatry() -> pd.DataFrame:
    # Load all physical visits data
    view = "[FOR_besoeg_fysiske_fremmoeder_inkl_2021_feb2022]"
    cols_to_keep = "datotid_start, datotid_slut, dw_ek_borger, psykambbesoeg AS pt_type"

    sql = "SELECT " + cols_to_keep + " FROM [fct]." + view
    sql += "WHERE datotid_start > '2012-01-01' AND psykambbesoeg = 1"

    df = pd.DataFrame(sql_load(sql, n_rows=None))  # type: ignore

    df[["datotid_start", "datotid_slut"]] = df[["datotid_start", "datotid_slut"]].apply(
        pd.to_datetime
    )

    # Subtract 1 day from datotid_slut in ambulant dates because we want to make predictions one day prior to visit
    df["datotid_predict"] = df["datotid_start"] - timedelta(days=1)  # type: ignore

    df = df.drop_duplicates(subset=["dw_ek_borger", "datotid_predict"])

    return df[["dw_ek_borger", "datotid_predict"]]  # type: ignore


def outpatient_visits_timestamps() -> pd.DataFrame:
    # Load forced_admissions data
    view = "[all_outpatient_visits_processed_2012_2021]"

    sql = "SELECT * FROM [fct]." + view

    outpatient_visits = pd.DataFrame(sql_load(sql, chunksize=None))  # type: ignore

    outpatient_visits = outpatient_visits.rename(columns={"datotid_predict": "timestamp"})

    return outpatient_visits


if __name__ == "__main__":
    get_outpatient_visits_to_psychiatry()
