"""Loaders for medications."""

from __future__ import annotations

import logging

import pandas as pd

from psycop.common.feature_generation.loaders_2025.diagnoses import load_from_codes

log = logging.getLogger(__name__)


def load(
    atc_code: str | list[str],
    output_col_name: str | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    wildcard_code: bool = True,
    n_rows: int | None = None,
    exclude_atc_codes: list[str] | None = None,
    administration_route: str | None = None,
    administration_method: str | None = None,
    fixed_doses: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Load medications. Aggregates prescribed/administered if both true. If
    wildcard_atc_code, match from atc_code*. Aggregates all that match. Beware
    that data is incomplete prior to sep. 2016 for prescribed medications.

    Args:
        atc_code (str): ATC-code prefix to load. Matches atc_code_prefix*.
            Aggregates all.
        output_col_name (str, optional): Name of output_col_name. Contains 1 if
            atc_code matches atc_code_prefix, 0 if not.Defaults to
            {atc_code_prefix}_value.
        load_prescribed (bool, optional): Whether to load prescriptions. Defaults to
            False. Beware incomplete until sep 2016.
        load_administered (bool, optional): Whether to load administrations.
            Defaults to True.
        wildcard_code (bool, optional): Whether to match on atc_code* or
            atc_code.
        n_rows (int, optional): Number of rows to return. Defaults to None, in which case all rows are returned.
        exclude_atc_codes (list[str], optional): Drop rows if atc_code is a direct match to any of these. Defaults to None.
        administration_route (str, optional): Whether to subset by a specific administration route, e.g. 'OR', 'IM' or 'IV'. Only applicable for administered medication, not prescribed. Defaults to None.
        administration_method (str, optional): Whether to subset by method of administration, e.g. 'PN' or 'Fast'. Only applicable for administered medication, not prescribed. Defaults to None.
        fixed_doses ( tuple(int), optional): Whether to subset by specific doses. Doses are set as micrograms (e.g., 100 mg = 100000). Defaults to None which return all doses. Find standard dosage for medications on pro.medicin.dk.
    Returns:
        pd.DataFrame: Cols: dw_ek_borger, timestamp, {atc_code_prefix}_value = 1
    """

    if load_prescribed and any([administration_method, administration_route]):
        raise TypeError(
            "load() got an unexpected combination of arguments. When load_prescribed=True, administration_method and administration_route must be NoneType objects."
        )

    if load_prescribed:
        log.warning(
            "Beware, there are missing prescriptions until september 2016. "
            "Hereafter, data is complete. See the wiki (OBS: Medication) for more details."
        )

    df = pd.DataFrame()

    if load_prescribed and load_administered:
        n_rows = int(n_rows / 2) if n_rows else None

    if load_prescribed:
        df_medication_prescribed = load_from_codes(
            codes_to_match=atc_code,
            code_col_name="atc",
            source_timestamp_col_name="datotid_ordinationstart",
            view="CVD_T2D_Medicin_ordineret_marts_2025",
            output_col_name=output_col_name,
            match_with_wildcard=wildcard_code,
            n_rows=n_rows,
            exclude_codes=exclude_atc_codes,
            load_diagnoses=False,
            administration_route=administration_route,
            administration_method=administration_method,
            fixed_doses=fixed_doses,
        )

        df = pd.concat([df, df_medication_prescribed])

    if load_administered:
        df_medication_administered = load_from_codes(
            codes_to_match=atc_code,
            code_col_name="atc",
            source_timestamp_col_name="datotid_administration_start",
            view="CVD_T2D_Medicin_administreret_marts_2025",
            output_col_name=output_col_name,
            match_with_wildcard=wildcard_code,
            n_rows=n_rows,
            exclude_codes=exclude_atc_codes,
            load_diagnoses=False,
            administration_route=administration_route,
            administration_method=administration_method,
            fixed_doses=fixed_doses,
        )
        df = pd.concat([df, df_medication_administered])

    if output_col_name is None:
        output_col_name = "_".join(atc_code) if isinstance(atc_code, list) else atc_code

    df = df.rename(columns={output_col_name: "value"})

    return df.reset_index(drop=True).drop_duplicates(  # type: ignore
        subset=["dw_ek_borger", "timestamp", "value"], keep="first"
    )


def concat_medications(
    output_col_name: str, atc_code_prefixes: list[str], n_rows: int | None = None
) -> pd.DataFrame:
    """Aggregate multiple blood_sample_ids (typically NPU-codes) into one
    column.

    Args:
        output_col_name (str): Name for new column.  # noqa: DAR102
        atc_code_prefixes (list[str]): list of atc_codes.
        n_rows (int, optional): Number of atc_codes to aggregate. Defaults to None.

    Returns:
        pd.DataFrame
    """
    dfs = [
        load(atc_code=f"{id}", output_col_name=output_col_name, n_rows=n_rows)
        for id in atc_code_prefixes  # noqa
    ]

    return (
        pd.concat(dfs, axis=0)
        .drop_duplicates(subset=["dw_ek_borger", "timestamp", "value"], keep="first")
        .reset_index(drop=True)  # type: ignore
    )


def antidiabetics_2025(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="A10",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=True,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )
