"""Example of."""
from __future__ import annotations

import logging

import pandas as pd

from psycop_feature_generation.loaders.raw.sql_load import sql_load

log = logging.getLogger(__name__)


def str_to_sql_match_logic(
    code_to_match: str,
    code_sql_col_name: str,
    load_diagnoses: bool,
    match_with_wildcard: bool,
) -> str:
    """Generate SQL match logic from a single string.

    Args:
        code_to_match (list[str]): List of strings to match.
        code_sql_col_name (str): Name of the SQL column containing the codes.
        load_diagnoses (bool): Whether to load diagnoses or medications. Determines the logic. See calling function for more.
        match_with_wildcard (bool): Whether to match on icd_code* / atc_code* or only icd_code / atc_code.
    """
    if load_diagnoses:
        base_query = f"lower({code_sql_col_name}) LIKE '%{code_to_match.lower()}"
    else:
        base_query = f"lower({code_sql_col_name}) LIKE '{code_to_match.lower()}"

    if match_with_wildcard:
        return f"{base_query}%'"

    if load_diagnoses:
        return f"{base_query}' OR {base_query}#%'"

    return f"{base_query}'"


def list_to_sql_logic(
    codes_to_match: list[str],
    code_sql_col_name: str,
    load_diagnoses: bool,
    match_with_wildcard: bool,
):
    """Generate SQL match logic from a list of strings.

    Args:
        codes_to_match (list[str]): List of strings to match.
        code_sql_col_name (str): Name of the SQL column containing the codes.
        load_diagnoses (bool): Whether to load diagnoses or medications. Determines the logic. See calling function for more.
        match_with_wildcard (bool): Whether to match on icd_code* / atc_code* or only icd_code / atc_code.
    """
    match_col_sql_strings = []

    for code_str in codes_to_match:
        if load_diagnoses:
            base_query = f"lower({code_sql_col_name}) LIKE '%{code_str.lower()}"
        else:
            base_query = f"lower({code_sql_col_name}) LIKE '{code_str.lower()}"

        if match_with_wildcard:
            match_col_sql_strings.append(
                f"{base_query}%'",
            )
        else:
            # If the string is at the end of diagnosegruppestreng, it doesn't end with a hashtag
            match_col_sql_strings.append(f"{base_query}'")

            if load_diagnoses:
                # If the string is at the beginning of diagnosegruppestreng, it doesn't start with a hashtag
                match_col_sql_strings.append(
                    f"lower({code_sql_col_name}) LIKE '{code_str.lower()}#%'",
                )

    return " OR ".join(match_col_sql_strings)


def load_from_codes(
    codes_to_match: list[str] | str,
    load_diagnoses: bool,
    code_col_name: str,
    source_timestamp_col_name: str,
    view: str,
    output_col_name: str | None = None,
    match_with_wildcard: bool = True,
    n_rows: int | None = None,
    exclude_codes: list[str] | None = None,
    administration_route: str | None = None,
    administration_method: str | None = None,
    shak_location_col: str | None = None,
    shak_code: int | None = None,
    keep_code_col: bool = False,
    shak_sql_operator: str | None = None,
) -> pd.DataFrame:
    """Load the visits that have diagnoses that match icd_code or atc code from
    the beginning of their adiagnosekode or atc code string. Aggregates all
    that match.

    Args:
        codes_to_match (Union[list[str], str]): Substring(s) to match diagnoses or medications for.
            Diagnoses: Matches any diagnoses, whether a-diagnosis, b-diagnosis.
            Both: If a list is passed, will count as a match if any of the icd_codes or at codes in the list match.
        load_diagnoses (bool): Determines which mathing logic is employed. If True, will load diagnoses. If False, will load medications.
            Diagnoses must be able to split a string like this:
                A:DF431#+:ALFC3#B:DF329
            Which means that if match_with_wildcard is False, we must match on *icd_code# or *icd_code followed by nothing. If it's true, we can match on *icd_code*.
        code_col_name (str): Name of column containing either diagnosis (icd) or medication (atc) codes.
            Takes either 'diagnosegruppestreng' or 'atc' as input.
        source_timestamp_col_name (str): Name of the timestamp column in the SQL
            view.
        view (str): Name of the SQL view to load from.
        output_col_name (str, optional): Name of new column string. Defaults to
            None.
        match_with_wildcard (bool, optional): Whether to match on icd_code* / atc_code*.
            Defaults to true.
        n_rows: Number of rows to return. Defaults to None.
        exclude_codes (list[str], optional): Drop rows if their code is in this list. Defaults to None.
        administration_route (str, optional): Whether to subset by a specific administration route, e.g. 'OR', 'IM' or 'IV'. Defaults to None.
        administration_method (str, optional): Whether to subset by method of administration, e.g. 'PN' or 'Fast'. Defaults to None.
        shak_location_col (str, optional): Name of column containing shak code. Defaults to None. Combine with shak_code and shak_sql_operator.
        shak_code (int, optional): Shak code indicating where to keep/not keep visits from (e.g. 6600). Defaults to None.
        keep_code_col (bool, optional): Whether to keep the code column. Defaults to False.
        shak_sql_operator (str, optional): Operator indicating how to filter shak_code, e.g. "!= 6600" or "= 6600". Defaults to None.

    Returns:
        pd.DataFrame: A pandas dataframe with dw_ek_borger, timestamp and
            output_col_name = 1
    """
    fct = f"[{view}]"

    if isinstance(codes_to_match, list) and len(codes_to_match) > 1:
        match_col_sql_str = list_to_sql_logic(
            codes_to_match=codes_to_match,
            code_sql_col_name=code_col_name,
            load_diagnoses=load_diagnoses,
            match_with_wildcard=match_with_wildcard,
        )
    elif isinstance(codes_to_match, str):
        match_col_sql_str = str_to_sql_match_logic(
            code_to_match=codes_to_match,
            code_sql_col_name=code_col_name,
            load_diagnoses=load_diagnoses,
            match_with_wildcard=match_with_wildcard,
        )
    else:
        raise ValueError("codes_to_match must be either a list or a string.")

    sql = (
        f"SELECT dw_ek_borger, {source_timestamp_col_name}, {code_col_name} "
        + f"FROM [fct].{fct} WHERE {source_timestamp_col_name} IS NOT NULL AND ({match_col_sql_str})"
    )

    if shak_code is not None:
        sql += f" AND left({shak_location_col}, {len(str(shak_code))}) {shak_sql_operator} {str(shak_code)}"

    if administration_method:
        allowed_administration_methods = (
            "Fast",
            "PN",
            "Engangs",
            "Alternerende",
            "Kontinuerlig",
            "Skema2",
            "Skema",
            "Tidspunkter",
        )
        if administration_method not in allowed_administration_methods:
            log.warning(
                "Value for administration method does not exist, returning 0 rows. "
                + f"Allowed values are {allowed_administration_methods}.",
            )
        sql += f" AND type_kodetekst = '{administration_method}'"

    if administration_route:
        allowed_administration_routes = (
            "OR",
            "IV",
            "IH",
            "IM",
            "SC",
            "KU",
            "IA",
            "IR",
            "PR",
            "IN",
            "OK",
            "TD",
            "PO",
            "SL",
            "BS",
            "DE",
            "ED",
            "CO",
            "VA",
            "LO",
            "PE",
            "AU",
            "RE",
            "IE",
            "IU",
            "UR",
            "GA",
            "OG",
            "OS",
            "IC",
            "OM",
            "ET",
            "HE",
            "IB",
            "BR",
            "KO",
            "VI",
            "EC",
            "IL",
            "IP",
            "IT",
            "MP",
            "CA",
            "IO",
            "IS",
            "CE",
            "ID",
            "ES",
            "SM",
            "TR",
            "PA",
            "PT",
            "TO",
            "PD",
            "ON",
            "BU",
            "GI",
            "OF",
            "AM",
            "CB",
            "EL",
            "PV",
            "LY",
            "XA",
            "IF",
            "AL",
            "DI",
            "PN",
            "PC",
            "BD",
            "IG",
            "HS",
            "TU",
            "PJ",
            "LF",
        )
        if administration_route not in allowed_administration_routes:
            log.warning(
                "Value for administration route does not exist, returning 0 rows. "
                + f"Allowed values are {allowed_administration_routes}.",
            )
        sql += f" AND admvej_kodetekst = '{administration_route}'"

    df = sql_load(sql, database="USR_PS_FORSK", chunksize=None, n_rows=n_rows)

    if exclude_codes:
        # Drop all rows whose code_col_name is in exclude_codes
        df = df[~df[code_col_name].isin(exclude_codes)]

    if output_col_name is None:
        if isinstance(codes_to_match, list):
            output_col_name = "_".join(codes_to_match)
        else:
            output_col_name = codes_to_match

    df[output_col_name] = 1

    if not keep_code_col:
        df = df.drop([f"{code_col_name}"], axis="columns")

    return df.rename(
        columns={
            source_timestamp_col_name: "timestamp",
        },
    )


def unpack_intervals(
    df: pd.DataFrame,
    starttime_col: str = "datotid_start_sei",
    endtime_col: str = "timestamp",
    entity_id: str = "dw_ek_borger",
    unpack_freq: str = "D",
) -> pd.DataFrame:
    """Transform df with starttime_col and endtime_col to day grain (one row per day in the interval starttime_col-endtime_col).
    First and last day will have the specific start and end time, while days inbetween will be 00:00:00.

    Args:
        df (pd.DataFrame): dataframe with time interval in separate columns.
        starttime_col (str, optional): Name of column with start time. Defaults to "datotid_start_sei".
        endtime_col (str, optional): Name of column with end time. Defaults to "timestamp".
        entity_id (str, optional): Name of column with entity id. Defaults to "dw_ek_borger".
        unpack_freq: Frequency string by which the interval will be unpacked. Default to "D" (day). For e.g., 5 hours, write "5H".

    Returns:
        pd.DataFrame: Dataframe with time interval unpacked to day grain.

    """

    # create rows with end time
    df_end_rows = df.copy()
    df_end_rows["date_range"] = df_end_rows[f"{endtime_col}"]

    # create a date range column between start date and end date for each visit/admission/coercion instance
    df["date_range"] = df.apply(
        lambda x: pd.date_range(
            start=x[f"{starttime_col}"],
            end=x[f"{endtime_col}"],
            freq=unpack_freq,
        ),
        axis=1,
    )

    # explode the date range column to create a new row for each date in the range
    df = df.explode("date_range")

    # concat df with start and end time rows
    df = pd.concat([df, df_end_rows], ignore_index=True).sort_values(
        [f"{entity_id}", f"{starttime_col}", "date_range"],
    )

    # drop duplicates (when start and/or end time = 00:00:00)
    df = df.drop_duplicates(keep="first")

    # reset index
    df = df.reset_index(drop=True)

    # set value to 1 (duration has lost meaning now, since durations are repeated on multiple rows per coercion instance now)
    df["value"] = 1

    # only keep relevant columns and rename date_range to timestamp
    df = df[[f"{entity_id}", "date_range", "value"]].rename(
        columns={"date_range": "timestamp"},
    )

    return df
