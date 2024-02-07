"""Utility functions for cleaning and repairing data breaks and errors in the
 raw contact and coercion data
 """
from datetime import timedelta

import pandas as pd


def lpr2_lpr3_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Function for fusing admissions split by LPR2>LPR3 transition. The change
    from LPR2 to LPR3 was made on 2019-02-03 01:00:00 All patients admitted at
    that time were 'discharged' and readmitted. Admissions crossing this period
    were split into two rows. Find these rows and merge them (keeping only one
    kontakt_id)

    Args:
        df (pd.DataFrame): Contact data

    Returns:
        df (pd.DataFrame): Contact data after processing
    """

    # Finding all admissions ending exactly on the specific timestamp (LPR2 IDs)
    df_LPR2 = df[df["datotid_slut"] == "2019-02-03 01:00:00"]

    # Removing column with discharge time from this df
    df_LPR2 = df_LPR2.drop(columns=["datotid_slut"])

    # Finding all admissions starting exactly on the specific timestamp (LPR3 IDs)
    df_LPR3 = df[df["datotid_start"] == "2019-02-03 01:00:00"]

    # Removing column with admission start times from this df
    df_LPR3 = df_LPR3.drop(columns=["datotid_start"])

    # Some patients are actually discharged or admitted at the given timestamps. Thus, only keep patients who are in both
    # only keep patients who occur across the change in LPR2 and LPR3
    df_LPR2 = df_LPR2[df_LPR2["dw_ek_borger"].isin(df_LPR3["dw_ek_borger"])]
    df_LPR3 = df_LPR3[df_LPR3["dw_ek_borger"].isin(df_LPR2["dw_ek_borger"])]

    # Remove ambulant duplicates from LPR2
    df_LPR2 = df_LPR2.sort_values(["dw_ek_borger", "pt_type"]).drop_duplicates(
        subset="dw_ek_borger", keep="first"
    )

    # Sorting values so that the order of the indexes corresponds to the ones of the LPR3 dataframe
    df_LPR2 = df_LPR2.sort_values(["dw_ek_borger"]).reset_index().drop(columns=["index"])

    df_LPR3 = df_LPR3.sort_values(["dw_ek_borger"]).reset_index().drop(columns=["index"])

    # Keep LPR2 patient IDs:
    # Creat temporary list of all discharge times from LPR3 version
    temp = df_LPR3["datotid_slut"].to_list()

    # Setting these values as discharge times for LPR2 version
    # (any new data reported in the LPR3 section of the admissions might be lost)
    df_LPR2.insert(2, "datotid_slut", temp)

    # Remove all the affected admissions from original df
    df_rest = df[df["datotid_slut"] != "2019-02-03 01:00:00"]
    df_rest = df_rest[df_rest["datotid_start"] != "2019-02-03 01:00:00"]

    # Concatenate the two dataframes
    df = pd.concat([df_rest, df_LPR2]).reset_index().drop(columns=["index"])

    return df


def concat_readmissions_for_all_patients(
    df: pd.DataFrame, readmission_interval_hours: int = 4
) -> pd.DataFrame:
    """
    Concatenates close  individual readmissions into continuous admissions.

    Args:
        df (pd.DateFrame): A data frame containing an ID column, an admission time column and a discharge time column for the admissions for all patients.
        readmission_interval_hours (int): Number of hours between admissions determining whether an admission is considered a readmission. Defaults to 4, following
        advice from the Danish Health Data Authority.

    Returns:
        pd.DateFrame: A data frame containing an ID column, and admission time column and a discharge time column for all admissions of a
        unique ID with readmissions concatenated.
    """

    df = df.sort_values(["dw_ek_borger", "datotid_start"])
    df_grouped = df.groupby("dw_ek_borger")

    # Get list of dfs; one for each patient
    df_patients_list = [df_grouped.get_group(key) for key in df_grouped.groups]

    # concatenate dataframes for individual patients
    df = pd.concat(
        [
            concat_readmissions_for_single_patient(
                patient,  # type: ignore
                readmission_interval_hours=readmission_interval_hours,
            )
            for patient in df_patients_list
        ]
    )

    return df


def concat_readmissions_for_single_patient(
    df_patient: pd.DataFrame, readmission_interval_hours: int = 4
) -> pd.DataFrame:
    """
    Concatenates individual readmissions into continuous admissions. An admission is defined as a readmission when the admission starts less than a specifiec number of
    hours after the previous admission. According to the Danish Health Data Authority, subsequent admissions should be regarded as one continious admission if the interval
    between admissions is less than four hours.
    In the case of multiple subsequent readmissions, all readmissions are concatenated into one admission.

    Args:
        df_patient (pd.DateFrame): A data frame containing an ID column, an admission time column and a discharge time column for the admissions of a unique ID.
        readmission_interval_hours (int): Number of hours between admissions determining whether an admission is considered a readmission. Defaults to 4, following
        advice from the Danish Health Data Authority.

    Returns:
        pd.DateFrame: A data frame containing an ID column, and admission time column and a discharge time column for all admissions of a
        unique ID with readmissions concatenated.
    """

    # 'end_readmission' indicates whether the end of the admission was followed be a readmission less than four hours later
    df_patient = df_patient.assign(
        end_readmission=lambda x: x["datotid_start"].shift(-1) - x["datotid_slut"]
        < pd.Timedelta(readmission_interval_hours, "hours")
    )
    # 'start_readmission' indicates whether the admission started less than four hours later after the previous admission
    df_patient = df_patient.assign(
        start_readmission=lambda x: x["datotid_start"] - x["datotid_slut"].shift(1)
        < pd.Timedelta(readmission_interval_hours, "hours")
    )

    # if the patients have any readmissions, the affected rows are subsetted
    if df_patient["end_readmission"].any() & df_patient["start_readmission"].any():
        readmissions = df_patient[
            (df_patient["end_readmission"] is True) | (df_patient["start_readmission"] is True)
        ]

        # if there are multiple subsequent readmissions (i.e., both 'end_readmission' and 'start_readmission' == True), all but the first and last are excluded
        readmissions_subset = readmissions[
            (df_patient["end_readmission"] is False) | (df_patient["start_readmission"] is False)
        ]

        # insert discharge time from the last readmission into the first
        readmissions_subset.loc[
            readmissions_subset["start_readmission"] is False, "datotid_slut"
        ] = readmissions_subset["datotid_slut"].shift(-1)

        # keep only the first admission
        readmissions_subset = readmissions_subset[readmissions_subset["end_readmission"] is True]

        # remove readmissions from the original data
        df_patient_no_readmissions = df_patient.merge(
            readmissions[["dw_ek_borger", "datotid_start"]],
            how="outer",
            on=["dw_ek_borger", "datotid_start"],
            indicator=True,
        )
        df_patient_no_readmissions = df_patient_no_readmissions.loc[
            df_patient_no_readmissions["_merge"] != "both"
        ]

        # merge the new rows with the rest of the admissions
        df_patient_concatenated_readmissions = df_patient_no_readmissions.merge(
            readmissions_subset, how="outer", on=["dw_ek_borger", "datotid_start", "datotid_slut"]
        )

    else:
        return df_patient[["dw_ek_borger", "datotid_start", "datotid_slut"]].sort_values(
            ["dw_ek_borger", "datotid_start"]
        )

    return df_patient_concatenated_readmissions[
        ["dw_ek_borger", "datotid_start", "datotid_slut"]
    ].sort_values(["dw_ek_borger", "datotid_start"])


def map_forced_admissions(
    forced_admissions: pd.DataFrame, admissions_df: pd.DataFrame
) -> pd.DataFrame:
    """Function for generating new contact id column in forced_admissions data
    that may match contacts data.

    Args:
        forced_admissions (pd.DataFrame): Forced admissions data from SEI
        admissions_df (pd.DataFrame): All admissions

    Returns:
        forced_admissions (pd.DataFrame): Forced admissions
    """
    # Create new columns in force data for admission start and end times
    forced_admissions["datotid_start"] = -1
    forced_admissions["datotid_slut"] = -1

    # Create empty list to hold the dataframes for individual patients
    df_list = []

    # Find the admissions the correspond to forced admissions by looking at start of admissions for each unique patient
    for patient in forced_admissions.dw_ek_borger.unique():
        # Subset dfs to specific patient
        one_patient_contacts = (
            admissions_df[admissions_df.dw_ek_borger == patient]
            .sort_values(by="datotid_start", ascending=True)
            .reset_index(drop=True)
        )

        one_patient_forced_admissions = (
            forced_admissions[forced_admissions.dw_ek_borger == patient]
            .sort_values(by="datotid_start_sei", ascending=True)
            .reset_index(drop=True)
        )

        # Loop through patient admission data
        for i, _ in one_patient_contacts.iterrows():
            # Loop through the patients forced admissions
            for j, _ in one_patient_forced_admissions.iterrows():
                # If start of forced admission is after start of admission timeand before next admissions
                # Add 23h amd 59m because all forced admissions in the data start at 00:00:00
                if (
                    (one_patient_forced_admissions["datotid_start"][j] == -1)
                    # If patient has two admissions that match a forced admissions (e.g. two admissions that start on the same day and one is a forced admissions)
                    # then by default label the first one as the forced admission. Often, the second admission will be a transfer to another ward with the patient now agreeing to be voluntary admitted
                    and (
                        one_patient_forced_admissions["datotid_start_sei"][j]
                        + timedelta(hours=23, minutes=59)
                        >= one_patient_contacts["datotid_start"][i]
                    )
                    and (
                        one_patient_forced_admissions["datotid_slut_sei"][j]
                        <= one_patient_contacts["datotid_slut"][i]
                    )
                ):
                    # Set start of admission
                    one_patient_forced_admissions["datotid_start"][j] = one_patient_contacts[
                        "datotid_start"
                    ][i]
                    # And set end of admission
                    one_patient_forced_admissions["datotid_slut"][j] = one_patient_contacts[
                        "datotid_slut"
                    ][i]

        # Append patient df to list
        df_list.append(one_patient_forced_admissions)

    # Concatenate dfs back into full df
    forced_admissions = pd.concat(df_list)

    forced_admissions = forced_admissions[forced_admissions["datotid_start"] != -1]
    forced_admissions = forced_admissions[forced_admissions["datotid_slut"] != -1]
    # 180 forced admissions dropped

    forced_admissions[["datotid_start", "datotid_slut"]] = forced_admissions[
        ["datotid_start", "datotid_slut"]
    ].apply(pd.to_datetime)

    # Keep only forced admissions that correspond to admissions in the cleaned admission data set
    forced_admissions.merge(admissions_df, on=["datotid_start", "dw_ek_borger"], how="inner")

    unmatching_start_dates = forced_admissions[
        pd.to_datetime(forced_admissions["datotid_start_sei"]).dt.date
        != pd.to_datetime(forced_admissions["datotid_start"]).dt.date
    ]
    print(
        len(unmatching_start_dates)
    )  # 346 'forced amdissions' with a different start date than the one registered in the SEI
    print(unmatching_start_dates.head(10))
    unmatching_end_dates = forced_admissions[
        pd.to_datetime(forced_admissions["datotid_slut_sei"]).dt.date
        != pd.to_datetime(forced_admissions["datotid_slut"]).dt.date
    ]
    print(
        len(unmatching_end_dates)
    )  # 3845 'forced amdissions' with a different end date than the one registered in the SEI
    print(unmatching_end_dates.head(10))
    unmatching_dates = unmatching_end_dates[
        pd.to_datetime(unmatching_end_dates["datotid_start_sei"]).dt.date
        != pd.to_datetime(unmatching_end_dates["datotid_start"]).dt.date
    ]
    print(
        len(unmatching_dates)
    )  # At least 188 'forced amdissions' have both a different start and end date than the SEI registration
    print(unmatching_dates.head(10))

    return (
        forced_admissions[["dw_ek_borger", "datotid_start", "datotid_slut"]]
        .reset_index()
        .drop(columns=["index"])
    )
