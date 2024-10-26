"""
Script for obtaining and writing all ambulant psychiatric visits. Handles LPR2 to LPR3 transition and duplicates
"""
import pandas as pd
import polars as pl

from datetime import timedelta

from psycop.common.feature_generation.loaders.raw.load_visits import ambulatory_visits

def get_outpatient_visits_to_psychiatry(write: bool = False) -> pd.DataFrame:
    # Load all physical visits data
    prediction_times = pl.from_pandas(
            ambulatory_visits(
                timestamps_only=True,
                timestamp_for_output="start",
                n_rows=None,
                return_value_as_visit_length_days=False,
                shak_code=6600,
                shak_sql_operator="=",
            )
        ).with_columns(pl.col("timestamp") - pl.duration(days=1)) #jeg forudser dagen før et ambulant besøg så man har oplysningen når pt kommer

    return prediction_times[["dw_ek_borger", "timestamp"]]  # type: ignore


if __name__ == "__main__":
    df_pl = get_outpatient_visits_to_psychiatry()
    df_pd = df_pl.to_pandas()
    n_patients = df_pd['dw_ek_borger'].nunique()
    print(f"Antal unikke ID'er der har mindst én ambulant kontakt er: {n_patients}")
    antal_kontakter = df_pd.shape[0]
    print(f"Antal ambulante kontakter: {antal_kontakter}")