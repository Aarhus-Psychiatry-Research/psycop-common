import pandas as pd
from psycop.common.feature_generation.loaders.raw import sql_load
from wasabi import msg


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
        f"SELECT * FROM [fct].[psycop_fa_outcome_all_disorders_forced_admission_{visit_type}_730.5d_0f_182d_2014_2021]",
        database="USR_PS_FORSK",
    )

    df = df[
        [
            "dw_ek_borger",
            prediction_times_col_name,
            "outc_182_days",
            "outcome_timestamp",
        ]
    ]

    df = df.rename(
        columns={
            prediction_times_col_name: "timestamp",
            "outc_182_days": "outc_bool_within_182_days",
            "outcome_timestamp": "outc_timestamp",
        },
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["outc_timestamp"] = pd.to_datetime(df["outc_timestamp"])

    if timestamps_only:
        df = df[["dw_ek_borger", "timestamp"]]

    return df
