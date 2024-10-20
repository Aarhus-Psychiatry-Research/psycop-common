"""Loaders for medications."""

from __future__ import annotations

import logging

import pandas as pd

from psycop.common.feature_generation.loaders.raw.utils import (
    str_to_sql_match_logic,
    list_to_sql_logic,
)
from psycop.common.feature_generation.loaders.raw.sql_load import sql_load

log = logging.getLogger(__name__)

def load_administered_med_from_codes(
    codes_to_match: list[str] | str,
    code_col_name: str,
    source_timestamp_col_name: str,
    output_col_name: str | None = None,
    cols_to_load: str = "dw_ek_borger, datotid_administration_start, antal, dosis, laegemiddelnavn, laegemiddelform_tekst, styrke_numerisk, styrke_enhed, type_kodetekst, admvej_kodetekst, afsnit_administration, atc",
    match_with_wildcard: bool = True,
    n_rows: int | None = None,
    exclude_codes: list[str] | None = None,
    administration_route: str | None = None,
    administration_method: str | None = None,
    fixed_doses: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Load the visits that have diagnoses that match icd_code or atc code from
    the beginning of their adiagnosekode or atc code string. Aggregates all
    that match.

    Args:
        codes_to_match (Union[list[str], str]): Substring(s) to match diagnoses or medications for.
            Diagnoses: Matches any diagnoses, whether a-diagnosis, b-diagnosis.
            Both: If a list is passed, will count as a match if any of the icd_codes or at codes in the list match.
        code_col_name (str): Name of column containing either diagnosis (icd) or medication (atc) codes.
            Takes either 'diagnosegruppestreng' or 'atc' as input.
        source_timestamp_col_name (str): Name of the timestamp column in the SQL
            view.
        output_col_name (str, optional): Name of output_col_name. Contains 1 if
            atc_code matches atc_code_prefix, 0 if not.Defaults to
            {atc_code_prefix}_value.
        output_col_name (str, optional): Name of new column string. Defaults to
            None.
        match_with_wildcard (bool, optional): Whether to match on icd_code* / atc_code*.
            Defaults to true.
        n_rows: Number of rows to return. Defaults to None.
        exclude_codes (list[str], optional): Drop rows if their code is in this list. Defaults to None.
        administration_route (str, optional): Whether to subset by a specific administration route, e.g. 'OR', 'IM' or 'IV'. Defaults to None.
        administration_method (str, optional): Whether to subset by method of administration, e.g. 'PN' or 'Fast'. Defaults to None.
        fixed_doses ( tuple(int), optional): Whether to subset by specific doses. Doses are set as micrograms (e.g., 100 mg = 100000). Defaults to None which return all doses. Find standard dosage for medications on pro.medicin.dk.

    Returns:
        pd.DataFrame: A pandas dataframe with dw_ek_borger, timestamp and
            output_col_name = 1
    """
    fct = "[FOR_Medicin_administreret_inkl_2021_feb2022]"

    match codes_to_match:
        case str():
            match_col_sql_str = str_to_sql_match_logic(
                code_to_match=codes_to_match,
                code_sql_col_name=code_col_name,
                load_diagnoses=False,
                match_with_wildcard=match_with_wildcard,
            )
        case list() if len(codes_to_match) == 1:
            match_col_sql_str = str_to_sql_match_logic(
                code_to_match=codes_to_match[0],
                code_sql_col_name=code_col_name,
                load_diagnoses=False,
                match_with_wildcard=match_with_wildcard,
            )
        case [*codes_to_match] if len(codes_to_match) > 1:
            match_col_sql_str = list_to_sql_logic(
                codes_to_match=codes_to_match,
                code_sql_col_name=code_col_name,
                load_diagnoses=False,
                match_with_wildcard=match_with_wildcard,
            )
        case list():
            raise ValueError("List is neither of len==1 or len>1")

    sql = (
        f"SELECT {cols_to_load} FROM [fct].{fct} WHERE {source_timestamp_col_name} IS NOT NULL AND ({match_col_sql_str})"
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

    if exclude_codes:
        # Drop all rows whose code_col_name is in exclude_codes
        df = df[~df[code_col_name].isin(exclude_codes)]

    if output_col_name is None:
        if isinstance(codes_to_match, list):
            output_col_name = "_".join(codes_to_match)
        else:
            output_col_name = codes_to_match

    df[output_col_name] = 1

    df = df.rename(columns={output_col_name: "value", source_timestamp_col_name: "timestamp"})

    return df


def uti_relevant_antibiotics(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    """Load all administered UIT-relevant (non-preventative) antibiotic medication"""
    return load_administered_med_from_codes(
        codes_to_match="J01",
        code_col_name="atc",
        source_timestamp_col_name="datotid_ordinationstart",
        match_with_wildcard=True,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )

if __name__ == "__main__":
    uti_relevant_antibiotics()