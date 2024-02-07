import pandas as pd
import polars as pl


def concat_readmissions(
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
        readmissions = df_patient[df_patient["end_readmission"] | df_patient["start_readmission"]]

        # if there are multiple subsequent readmissions (i.e., both 'end_readmission' and 'start_readmission' == True), all but the first and last are excluded
        readmissions_subset = readmissions[
            (readmissions.end_readmission == False) # type: ignore
            | (readmissions.start_readmission == False) # type: ignore
        ]

        # insert discharge time from the last readmission into the first
        readmissions_subset.loc[
            readmissions_subset.start_readmission == False, # type: ignore
            "datotid_slut",
        ] = readmissions_subset["datotid_slut"].shift(-1)

        # keep only the first admission
        readmissions_subset = readmissions_subset[
            readmissions_subset.end_readmission == True # type: ignore
        ]

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
            readmissions_subset,
            how="outer",
            on=["dw_ek_borger", "datotid_start", "datotid_slut", "shakkode_ansvarlig"],
        )

    else:
        return df_patient[
            ["dw_ek_borger", "datotid_start", "datotid_slut", "shakkode_ansvarlig"]
        ].sort_values(["dw_ek_borger", "datotid_start"])

    return df_patient_concatenated_readmissions[
        ["dw_ek_borger", "datotid_start", "datotid_slut", "shakkode_ansvarlig"]
    ].sort_values(["dw_ek_borger", "datotid_start"])


def preprocess_readmissions(df: pl.DataFrame) -> pl.DataFrame:
    df_ = df.to_pandas()

    df_ = df_.sort_values(["dw_ek_borger", "datotid_start"])

    df_patients = df_.groupby("dw_ek_borger")

    df_patients_list = [df_patients.get_group(key) for key in df_patients.groups]

    df = pd.concat([concat_readmissions(patient) for patient in df_patients_list])  # type: ignore

    return pl.DataFrame(df)


def keep_first_coercion_within_admission(admission: pd.DataFrame) -> pd.DataFrame:
    """Find first coercion instances within admissions

    Args:
        admission (pd.DataFrame): Dataframe including admissions on admission grain

    Returns:
        pd.DataFrame: Dataframe with added columns showing first occurance of mechanical, chemical, or physical restraint
    """
    admission = admission.assign(
        first_mechanical_restraint=admission.datotid_start_sei[
            admission.typetekst_sei == "Bælte"
        ].min()
    )
    admission = admission.assign(
        first_forced_medication=admission.datotid_start_sei[
            admission.typetekst_sei == "Beroligende medicin"
        ].min()
    )
    admission = admission.assign(
        first_manual_restraint=admission.datotid_start_sei[
            admission.typetekst_sei == "Fastholden"
        ].min()
    )

    return admission.drop(columns="typetekst_sei")[
        (admission.datotid_start_sei == admission.datotid_start_sei.min())
    ].drop_duplicates()


def select_outcomes(df: pl.DataFrame) -> pl.DataFrame:
    df_ = df.to_pandas()
    groups = df_.groupby(["dw_ek_borger", "datotid_start"])
    df_list = [groups.get_group(key[0]) for key in groups]

    df_concat = pd.concat(
        [keep_first_coercion_within_admission(admission) for admission in df_list],
    )

    return pl.DataFrame(df_concat)


def unpack_adm_days(
    idx: int,
    row: pd.Series,
    pred_hour: int = 6,
) -> pd.DataFrame:
    """Unpack admissions to long format (one row per day in the admission)

    Args:
        idx (int): row index
        row (pd.DataFrame): one admission
        pred_hour (int): prediction hour. Defaults to 6.

    Returns:
        pd.DataFrame: _description_
    """

    row = pd.DataFrame(row).transpose()  # type: ignore

    # expand admission days between admission start and discharge
    adm_day = pd.DataFrame(
        pd.date_range(row.loc[idx, "datotid_start"].date(), row.loc[idx, "datotid_slut"].date())
    )

    # add admission start to every day of admission
    adm_day["datotid_start"] = row.loc[idx, "datotid_start"]

    # join adm_day with row
    days_unpacked = pd.merge(row, adm_day, how="left", on="datotid_start")

    # add counter for days
    days_unpacked["pred_adm_day_count"] = adm_day.groupby(by="datotid_start").cumcount() + 1

    # add prediction time to prediction dates
    days_unpacked = days_unpacked.assign(pred_time=lambda x: x[0] + pd.Timedelta(hours=pred_hour))

    # exclude admission start days where admission happens after prediction
    if days_unpacked.loc[0, "datotid_start"] >= days_unpacked.loc[0, "pred_time"]:  # type: ignore
        days_unpacked = days_unpacked.iloc[1:, :]

    # if admission is longer than 1 day and if time is <= pred_hour
    if (len(days_unpacked) > 1) and (
        days_unpacked.iloc[-1, 2].time() <= days_unpacked.iloc[-1, -1].time()  # type: ignore
    ):
        days_unpacked = days_unpacked.iloc[:-1, :]

    return days_unpacked.drop(columns=0)


def explode_admissions(df: pl.DataFrame) -> pl.DataFrame:
    df_ = df.to_pandas()

    df_concat = pd.concat([unpack_adm_days(idx, row) for idx, row in df_.iterrows()])  # type: ignore

    return pl.DataFrame(df_concat)
