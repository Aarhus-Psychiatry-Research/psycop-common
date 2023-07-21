"""Utility functions for cohort hyperparameters"""
import pandas as pd


def exclude_prior_outcome_with_lookbehind(
    df: pd.DataFrame,
    lookbehind: int = 365,
    col_admission_start: str = "datotid_start",
    col_outcome_start: str = "datotid_start_sei",
) -> pd.DataFrame:
    """Exclude admissions with the outcome within the lookbehind, i.e. between 0 and lookbehind days before admission start.
        If lookbehind = 365, admissions with start time between 0-365 days prior will be excluded.

    Args:
        df (pd.DataFrame): Dataframe containing admissions
        lookbehind (int, optional): Lookbehind window, if the outcome occurs in the time between 0 and the lookbehind prior to the start, this admission will be excluded. Defaults to 365.
        col_admission_start (str, optional): Name of column with start timestamp. Defaults to "datotid_start".
        col_outcome_start (str, optional): Name of column with outcome timestamp. Defaults to "datotid_start_sei".

    Returns:
        pd.DataFrame: Dataframe without the excluded admissions
    """
    df_exclude = df[
        (df[col_admission_start] - df[col_outcome_start] >= pd.Timedelta(0, "days"))
        & (df.datotid_start - df[col_outcome_start] <= pd.Timedelta(lookbehind, "days"))
    ]
    return df_exclude


def cut_off_prediction_days(df: pd.DataFrame, cut_off: pd.Timedelta) -> pd.DataFrame:
    """Cut off prediction days after cut off

    Args:
        df (pd.DataFrame): _description_
        cut_off (float): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # cut days after cut-off
    df_cohort_exclude_days_after_cut_off = df[
        (pd.to_timedelta(df["pred_adm_day_count"], "days") <= cut_off)
    ]

    return df_cohort_exclude_days_after_cut_off


def create_binary_labels_with_lookahead(
    df: pd.DataFrame,
    lookahead_days: int,
) -> pd.DataFrame:
    """Depending on the lookahead window, create label columns indicating
    - outcome_coercion_bool_within_{lookahead_days}_days (int): 0 or 1, depending on whether a coercion instance happened within the lookahead window.

    Args:
        df (pd.DataFrame): coercion cohort dataframe
        lookahead_days (int): How far to look for coercion instances in days

    Returns:
        pd.DateFrame: df with two added columns (outcome_coercion_bool_within_{lookahead_days}_days, and outcome_coercion_type_within_{lookahead_days}_days)
    """

    # Outcome bool
    df[f"outcome_coercion_bool_within_{lookahead_days}_days"] = 0
    df.loc[
        (df["diff_first_coercion"] < pd.Timedelta(f"{lookahead_days} days"))
        & (df["include_pred_time"] == 1),
        f"outcome_coercion_bool_within_{lookahead_days}_days",
    ] = 1

    return df


def create_binary_and_categorical_labels_with_lookahead(
    df: pd.DataFrame,
    lookahead_days: int,
) -> pd.DataFrame:
    """Depending on the lookahead window, create label columns indicating;
    - outcome_coercion_bool_within_{lookahead_days}_days (int): 0 or 1, depending on whether a coercion instance happened within the lookahead window.
    - outcome_coercion_type_within_{lookahead_days}_days (int): which coercion type happened within the lookahead window.

    If multiple coercion types happened within the lookahead window, the most "severe" coercion type will be chosen.
    Coercion hierarchy/classes (least to most severe coercion type):
    0) No coercion
    1) Manual restraint
    2) Forced medication
    3) Mechanial restraint

    Args:
        df (pd.DataFrame): coercion cohort dataframe
        lookahead_days (int): How far to look for coercion instances in days

    Returns:
        pd.DateFrame: df with three two columns (outcome_coercion_bool_within_{lookahead_days}_days, and outcome_coercion_type_within_{lookahead_days}_days)
    """

    # Outcome bool
    df[f"outcome_coercion_bool_within_{lookahead_days}_days"] = 0
    df.loc[
        (df["diff_first_coercion"] < pd.Timedelta(f"{lookahead_days} days"))
        & (df["include_pred_time"] == 1),
        f"outcome_coercion_bool_within_{lookahead_days}_days",
    ] = 1

    # Mechanical restraint bool
    df[f"outcome_mechanical_restraint_bool_within_{lookahead_days}_days"] = 0
    df.loc[
        (
            df["diff_first_mechanical_restraint"]
            < pd.Timedelta(f"{lookahead_days} days")
        ),
        f"outcome_mechanical_restraint_bool_within_{lookahead_days}_days",
    ] = 1

    df[f"outcome_chemical_restraint_bool_within_{lookahead_days}_days"] = 0
    df.loc[
        (df["diff_first_forced_medication"] < pd.Timedelta(f"{lookahead_days} days")),
        f"outcome_chemical_restraint_bool_within_{lookahead_days}_days",
    ] = 1

    df[f"outcome_manual_restraint_bool_within_{lookahead_days}_days"] = 0
    df.loc[
        (df["diff_first_manual_restraint"] < pd.Timedelta(f"{lookahead_days} days")),
        f"outcome_manual_restraint_bool_within_{lookahead_days}_days",
    ] = 1

    # Outcome type
    df[f"outcome_coercion_type_within_{lookahead_days}_days"] = 0

    # Mechanical restraint (3)
    df.loc[
        (df[f"outcome_coercion_bool_within_{lookahead_days}_days"] == 1)
        & (
            df["diff_first_mechanical_restraint"]
            < pd.Timedelta(f"{lookahead_days} days")
        ),
        f"outcome_coercion_type_within_{lookahead_days}_days",
    ] = 3

    # Chemical restraint (2)
    df.loc[
        (df[f"outcome_coercion_bool_within_{lookahead_days}_days"] == 1)
        & (df[f"outcome_coercion_type_within_{lookahead_days}_days"] != 3)
        & (df["diff_first_forced_medication"] < pd.Timedelta(f"{lookahead_days} days")),
        f"outcome_coercion_type_within_{lookahead_days}_days",
    ] = 2

    # Physical restraint (1)
    df.loc[
        (df[f"outcome_coercion_bool_within_{lookahead_days}_days"] == 1)
        & (df[f"outcome_coercion_type_within_{lookahead_days}_days"] != 2)
        & (df[f"outcome_coercion_type_within_{lookahead_days}_days"] != 3)
        & (df["diff_first_manual_restraint"] < pd.Timedelta(f"{lookahead_days} days")),
        f"outcome_coercion_type_within_{lookahead_days}_days",
    ] = 1

    return df
