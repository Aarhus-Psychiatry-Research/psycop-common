import pandas as pd
from wasabi import msg

from psycop.common.feature_generation.loaders.raw import sql_load


def forced_admissions_inpatient(timestamps_only: bool = True) -> pd.DataFrame:
    df = process_forced_pred_dfs(
        visit_type="inpatient",
        prediction_times_col_name="datotid_slut",
        timestamps_only=timestamps_only,
    )

    msg.good(
        "Finished loading data frame for forced admissions with prediction times and outcome for all inpatient admissions",
    )

    return df.reset_index(drop=True)


def forced_admissions_outpatient(
    timestamps_only: bool = True,
) -> pd.DataFrame:
    df = process_forced_pred_dfs(
        visit_type="outpatient",
        prediction_times_col_name="datotid_predict",
        timestamps_only=timestamps_only,
    )

    msg.good(
        "Finished loading data frame for forced admissions with prediction times and outcome for all outpatient visits",
    )

    return df.reset_index(drop=True)


def process_forced_pred_dfs(
    visit_type: str,
    prediction_times_col_name: str,
    timestamps_only: bool = True,
) -> pd.DataFrame:
    df = sql_load(
        f"SELECT * FROM [fct].[psycop_fa_outcome_all_disorders_forced_admission_{visit_type}_730.5d_0f_2014_2021]",
        database="USR_PS_FORSK",
    )

    df = df[
        [
            "dw_ek_borger",
            prediction_times_col_name,
            "outc_91_days",
            "outc_182_days",
            "outc_365_days",
            "outc_timestamp_91_days",
            "outc_timestamp_182_days",
            "outc_timestamp_365_days",
        ]
    ]

    df = df.rename(
        columns={
            prediction_times_col_name: "timestamp",
            "outc_91_days": "outc_bool_within_91days",
            "outc_182_days": "outc_bool_within_182_days",
            "outc_365_days": "outc_bool_within_365_days",
            "outc_timestamp_91_days": "timestamp_outcome_91_days",
            "outc_timestamp_182_days": "timestamp_outcome_182_days",
            "outc_timestamp_365_days": "timestamp_outcome_365_days",
        },
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp_outcome_91_days"] = pd.to_datetime(df["timestamp_outcome_91_days"])
    df["timestamp_outcome_182_days"] = pd.to_datetime(df["timestamp_outcome_182_days"])
    df["timestamp_outcome_365_days"] = pd.to_datetime(df["timestamp_outcome_365_days"])

    if timestamps_only:
        df = df[["dw_ek_borger", "timestamp"]]

    return df
