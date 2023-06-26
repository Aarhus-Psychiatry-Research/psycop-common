import polars as pl

from psycop.common.feature_generation.loaders.raw.load_visits import ambulatory_visits



def get_eligible_prediction_times_as_polars() -> pl.DataFrame:
    df = pl.from_pandas(
        ambulatory_visits(timestamps_only=True, timestamp_for_output="start", n_rows=None, return_value_as_visit_length_days=False, shak_code=6600, shak_sql_operator="=")
    )

    filtered_df = filter_prediction_times_by_scz_bp_eligibility(
        df=df
    ).select(["dw_ek_borger", "timestamp"])
    return filtered_df