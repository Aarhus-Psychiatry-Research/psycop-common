import pandas as pd
import polars as pl

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load


def load_admissions_discharge_timestamps() -> pd.DataFrame:
    return sql_load("SELECT * FROM fct.[FOR_kohorte_indhold_pt_journal_inkl_2021_feb2022]")


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
        end_readmission=lambda x: x["timestamp"].shift(-1) - x["datotid_slut"]
        < pd.Timedelta(readmission_interval_hours, "hours")
    )
    # 'start_readmission' indicates whether the admission started less than four hours later after the previous admission
    df_patient = df_patient.assign(
        start_readmission=lambda x: x["timestamp"] - x["datotid_slut"].shift(1)
        < pd.Timedelta(readmission_interval_hours, "hours")
    )

    # if the patients have any readmissions, the affected rows are subsetted
    if df_patient["end_readmission"].any() & df_patient["start_readmission"].any():
        readmissions = df_patient[df_patient["end_readmission"] | df_patient["start_readmission"]]

        # if there are multiple subsequent readmissions (i.e., both 'end_readmission' and 'start_readmission' == True), all but the first and last are excluded
        readmissions_subset = readmissions[
            (readmissions.end_readmission == False) | (readmissions.start_readmission == False)  # noqa E712
        ]

        # insert discharge time from the last readmission into the first
        readmissions_subset.loc[
            readmissions_subset.start_readmission == False, "datotid_slut"  # noqa E712
        ] = readmissions_subset["datotid_slut"].shift(-1)

        # keep only the first admission
        readmissions_subset = readmissions_subset[readmissions_subset.end_readmission == True]  # noqa E712

        # remove readmissions from the original data
        df_patient_no_readmissions = df_patient.merge(
            readmissions[["dw_ek_borger", "timestamp"]],
            how="outer",
            on=["dw_ek_borger", "timestamp"],
            indicator=True,
        )
        df_patient_no_readmissions = df_patient_no_readmissions.loc[
            df_patient_no_readmissions["_merge"] != "both"
        ]

        # merge the new rows with the rest of the admissions
        df_patient_concatenated_readmissions = df_patient_no_readmissions.merge(
            readmissions_subset,
            how="outer",
            on=["dw_ek_borger", "timestamp", "datotid_slut", "shakkode_ansvarlig"],
        )

    else:
        return df_patient[
            ["dw_ek_borger", "timestamp", "datotid_slut", "shakkode_ansvarlig"]
        ].sort_values(["dw_ek_borger", "timestamp"])

    return df_patient_concatenated_readmissions[
        ["dw_ek_borger", "timestamp", "datotid_slut", "shakkode_ansvarlig"]
    ].sort_values(["dw_ek_borger", "timestamp"])


def preprocess_readmissions(df: pl.DataFrame) -> pl.LazyFrame:
    df_ = df.to_pandas()

    df_ = df_.sort_values(["dw_ek_borger", "timestamp"])

    df_patients = df_.groupby("dw_ek_borger")

    df_patients_list = [df_patients.get_group(key) for key in df_patients.groups]

    df = pd.concat([concat_readmissions(patient) for patient in df_patients_list])  # type: ignore

    return pl.LazyFrame(df)


def unpack_adm_days(idx: int, row: pd.Series, pred_hour: int = 6) -> pd.DataFrame:  # type: ignore
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
        pd.date_range(row.loc[idx, "timestamp"].date(), row.loc[idx, "datotid_slut"].date())
    )

    # add admission start to every day of admission
    adm_day["timestamp"] = row.loc[idx, "timestamp"]

    # join adm_day with row
    days_unpacked = pd.merge(row, adm_day, how="left", on="timestamp")

    # add counter for days
    days_unpacked["pred_adm_day_count"] = adm_day.groupby(by="timestamp").cumcount() + 1

    # add prediction time to prediction dates
    days_unpacked = days_unpacked.assign(pred_time=lambda x: x[0] + pd.Timedelta(hours=pred_hour))

    # exclude admission start days where admission happens after prediction
    if days_unpacked.loc[0, "timestamp"] >= days_unpacked.loc[0, "pred_time"]:  # type: ignore
        days_unpacked = days_unpacked.iloc[1:, :]

    # if admission is longer than 1 day and if time is <= pred_hour
    if (len(days_unpacked) > 1) and (
        days_unpacked.iloc[-1, 2].time() <= days_unpacked.iloc[-1, -1].time()  # type: ignore
    ):
        days_unpacked = days_unpacked.iloc[:-1, :]

    return days_unpacked.drop(columns=0)


def explode_admissions(df: pl.LazyFrame) -> pl.LazyFrame:
    df_ = df.collect().to_pandas()

    df_concat = pd.concat([unpack_adm_days(idx, row) for idx, row in df_.iterrows()])  # type: ignore
    
    df_concat = df_concat[['dw_ek_borger', 'pred_time', 'pred_adm_day_count']]

    df_concat = df_concat.rename(columns={"pred_time": "timestamp"})
    
    return pl.LazyFrame(df_concat)
