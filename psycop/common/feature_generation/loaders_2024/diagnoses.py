"""Loaders for diagnosis codes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import pandas as pd

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.common.feature_generation.loaders.raw.utils import (
    list_to_sql_logic,
    str_to_sql_match_logic,
)
from psycop.common.feature_generation.loaders_2024.diabetes_filters import (
    keep_rows_where_diag_matches_t1d_diag,
    keep_rows_where_diag_matches_t2d_diag,
)

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)


def from_contacts(
    icd_code: list[str] | str,
    output_col_name: str = "value",
    code_col_name: str = "diagnosegruppestreng",
    n_rows: int | None = None,
    wildcard_icd_code: bool = False,
    keep_code_col: bool = False,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    """Load diagnoses from all hospital contacts. If icd_code is a list, will
    aggregate as one column (e.g. ["E780", "E785"] into a ypercholesterolemia
    column).

    Args:
        icd_code (str): Substring to match diagnoses for. Matches any diagnoses, whether a-diagnosis, b-diagnosis etc. # noqa: DAR102
        output_col_name (str, optional): Name of new column string. Defaults to "value".
        code_col_name (str, optional): Name of column in loaded data frame from which to extract the diagnosis codes. Defaults to "diagnosegruppestrengs".
        n_rows: Number of rows to return. Defaults to None.
        wildcard_icd_code (bool, optional): Whether to match on icd_code*. Defaults to False.
        keep_code_col (bool, optional): Whether to keep the code column. Defaults to False.
        timestamp_purpose (Literal[str], optional): The intended use of the loader. If used as a predictor, the timestamp should be set to the contact end time, in order to avoid data leakage from future
        timestamp_purpose (Literal[str], optional): The intended use of the loader. If used as a predictor, the timestamp should be set to the contact end time, in order to avoid data leakage from future
            events. If used a an outcome, the timestamp should be set as the contact start time, in order to avoid inflation of model performance.

    Returns:
        pd.DataFrame
    """

    log.warning(
        "The DNPR3 data model replaced the DNPR2 model on 3 February 2019. Due to changes in DNPR3 granularity of diagnoses differ across the two models. If your prediction timestamps, lookbehind or lookahead span across this date, you should not use count as a resolve_multiple_fn. See the wiki (LPR2 compared to LPR3) for more information."
    )

    log.warning(
        "Diagnoses should be identified by either contact start or end time, depending on whether the diagnoses are intended as predictors or outcomes. See the wiki (OBS: Diagnosis as outcome) for more information."
    )

    allowed_timestamp_purposes = ("predictor", "outcome")
    if timestamp_purpose not in allowed_timestamp_purposes:
        raise ValueError(
            f"Invalid value for timestamp_purpose. "
            f"Allowed values are {allowed_timestamp_purposes}."
        )

    if timestamp_purpose == "predictor":
        source_timestamp_col_name = "datotid_slut"
    elif timestamp_purpose == "outcome":
        source_timestamp_col_name = "datotid_start"

    df = load_from_codes(
        codes_to_match=icd_code,
        code_col_name=code_col_name,
        source_timestamp_col_name=source_timestamp_col_name,
        view="CVD_T2D_Kohorte_indhold",
        output_col_name=output_col_name,
        match_with_wildcard=wildcard_icd_code,
        n_rows=n_rows,
        load_diagnoses=True,
        keep_code_col=keep_code_col,
    )

    df = df.drop_duplicates(subset=["dw_ek_borger", "timestamp", output_col_name], keep="first")

    return df.reset_index(drop=True)  # type: ignore


def type_2_diabetes(
    n_rows: int | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    df = from_contacts(
        icd_code=[
            "E1",
            "E16",
            "O24",
            "T383A",
            "M142",
            "G590",
            "G632",
            "H280",
            "H334",
            "H360",
            "H450",
            "N083",
        ],
        wildcard_icd_code=True,
        n_rows=n_rows,
        timestamp_purpose=timestamp_purpose,
        keep_code_col=True,
    )

    df_filtered = keep_rows_where_diag_matches_t2d_diag(df=df, col_name="diagnosegruppestreng")

    return df_filtered.drop("diagnosegruppestreng", axis=1)


def type_1_diabetes(
    n_rows: int | None = None,
    timestamp_purpose: Literal["predictor", "outcome"] | None = "predictor",
) -> pd.DataFrame:
    df = from_contacts(
        icd_code=[
            "E1",
            "E16",
            "O24",
            "T383A",
            "M142",
            "G590",
            "G632",
            "H280",
            "H334",
            "H360",
            "H450",
            "N083",
        ],
        wildcard_icd_code=True,
        n_rows=n_rows,
        timestamp_purpose=timestamp_purpose,
        keep_code_col=True,
    )

    df_filtered = keep_rows_where_diag_matches_t1d_diag(df=df, col_name="diagnosegruppestreng")

    return df_filtered.drop("diagnosegruppestreng", axis=1)


def load_from_codes(
    codes_to_match: list[str] | str,
    load_diagnoses: bool,
    code_col_name: str,
    source_timestamp_col_name: str,
    view: str,
    output_col_name: str | None = None,
    match_with_wildcard: bool = True,
    n_rows: int | None = None,
    administration_route: str | None = None,
    administration_method: str | None = None,
    fixed_doses: tuple[int, ...] | None = None,
    keep_code_col: bool = False,
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
        administration_route (str, optional): Whether to subset by a specific administration route, e.g. 'OR', 'IM' or 'IV'. Defaults to None.
        administration_method (str, optional): Whether to subset by method of administration, e.g. 'PN' or 'Fast'. Defaults to None.
        fixed_doses ( tuple(int), optional): Whether to subset by specific doses. Doses are set as micrograms (e.g., 100 mg = 100000). Defaults to None which return all doses. Find standard dosage for medications on pro.medicin.dk.
        keep_code_col (bool, optional): Whether to keep the code column. Defaults to False.

    Returns:
        pd.DataFrame: A pandas dataframe with dw_ek_borger, timestamp and
            output_col_name = 1
    """
    fct = f"[{view}]"

    match codes_to_match:
        case str():
            match_col_sql_str = str_to_sql_match_logic(
                code_to_match=codes_to_match,
                code_sql_col_name=code_col_name,
                load_diagnoses=load_diagnoses,
                match_with_wildcard=match_with_wildcard,
            )
        case list() if len(codes_to_match) == 1:
            match_col_sql_str = str_to_sql_match_logic(
                code_to_match=codes_to_match[0],
                code_sql_col_name=code_col_name,
                load_diagnoses=load_diagnoses,
                match_with_wildcard=match_with_wildcard,
            )
        case [*codes_to_match] if len(codes_to_match) > 1:
            match_col_sql_str = list_to_sql_logic(
                codes_to_match=codes_to_match,
                code_sql_col_name=code_col_name,
                load_diagnoses=load_diagnoses,
                match_with_wildcard=match_with_wildcard,
            )
        case list():
            raise ValueError("List is neither of len==1 or len>1")

    sql = (
        f"SELECT dw_ek_borger, {source_timestamp_col_name}, {code_col_name} "
        + f"FROM [fct].{fct} WHERE {source_timestamp_col_name} IS NOT NULL AND ({match_col_sql_str})"
    )

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
                + f"Allowed values are {allowed_administration_methods}."
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
                + f"Allowed values are {allowed_administration_routes}."
            )
        sql += f" AND admvej_kodetekst = '{administration_route}'"

    if fixed_doses:
        sql += f" AND styrke_numerisk IN {fixed_doses}"

    df = sql_load(sql, database="USR_PS_FORSK", n_rows=n_rows)

    if output_col_name is None:
        if isinstance(codes_to_match, list):
            output_col_name = "_".join(codes_to_match)
        else:
            output_col_name = codes_to_match

    df[output_col_name] = 1

    if not keep_code_col:
        df = df.drop([f"{code_col_name}"], axis="columns")

    return df.rename(columns={source_timestamp_col_name: "timestamp"})
