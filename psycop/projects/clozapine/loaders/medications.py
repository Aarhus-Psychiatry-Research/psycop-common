"""Loaders for medications."""

from __future__ import annotations

import logging

import pandas as pd

from psycop.common.feature_generation.loaders_2024.diagnoses import load_from_codes

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
            view="Clozapin_medicin_ordineret",
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
            view="Clozapin_medicin_administreret",
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


def antipsychotics(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    """All antipsyhotics, except Lithium.

    Lithium is typically considered a mood stabilizer, not an
    antipsychotic.
    """
    return load(
        atc_code="N05A",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=True,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
        exclude_atc_codes=["N05AN01"],
    )


# 1. generation antipsychotics [flupentixol, pimozid, haloperidol, zuclopenthixol, melperon,pipamperon, chlorprotixen]


def first_gen_antipsychotics(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code=["N05AF01", "N05AG02", "N05AD01", "N05AF05", "N05AD03", "N05AD05", "N05AF03"],
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


# 2. generation antipsychotics [amisulpride, aripiprazole,asenapine, brexpiprazole, cariprazine, lurasidone, olanzapine, paliperidone, Quetiapine, risperidone, sertindol]


def second_gen_antipsychotics(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code=[
            "N05AL05",
            "N05AX12",
            "N05AH05",
            "N05AX16",
            "N05AX15",
            "N05AE02",
            "N05AE05",
            "N05AH03",
            "N05AX13",
            "N05AH04",
            "N05AX08",
            "N05AE04",
            "N05AE03",
        ],
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def top_10_weight_gaining_antipsychotics(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    """Top 10 weight gaining antipsychotics based on Huhn et al.

    2019. Only 5 of them are marketed in Denmark.
    """
    return load(
        atc_code=["N05AH03", "N05AE03", "N05AH04", "N05AX13", "N05AX08"],
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def sedative_antipsychotics(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code=["N05AH03", "N05AD01", "N05AF05", "N05AH04"],
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def non_sedative_antipsychotics(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N05A",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=True,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
        exclude_atc_codes=["N05AH03", "N05AD01", "N05AF05", "N05AH04"],
    )


def olanzapine_depot(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N05AH03",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route="IM",
        administration_method=administration_method,
        fixed_doses=(  # doses are found on pro.medicin.dk
            150000,
            210000,
            300000,
            405000,
        ),
    )


def aripiprazole_depot(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N05AX12",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route="IM",
        administration_method=administration_method,
        fixed_doses=(  # doses are found on pro.medicin.dk
            200000,
            300000,
            400000,
        ),
    )


def risperidone_depot(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N05AX08",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route="IM",
        administration_method=administration_method,
        fixed_doses=(  # doses are found on pro.medicin.dk
            25000,
            37500,
            50000,
        ),
    )


def paliperidone_depot(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N05AX13",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route="IM",
        administration_method=administration_method,
        fixed_doses=(  # doses are found on pro.medicin.dk
            25000,
            50000,
            75000,
            100000,
            150000,
            175000,
            263000,
            350000,
            525000,
        ),
    )


def haloperidol_depot(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N05AD01",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route="IM",
        administration_method=administration_method,
        fixed_doses=(  # doses are found on pro.medicin.dk
            50000,
            100000,
            150000,
            200000,
            250000,
            300000,
        ),
    )


def perphenazine_depot(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N05AB03",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route="IM",
        administration_method=administration_method,
        fixed_doses=(  # doses are found on pro.medicin.dk
            54000,
            108000,
            162000,
            216000,
        ),
    )


def zuclopenthixol_depot(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N05AF05",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route="IM",
        administration_method=administration_method,
        fixed_doses=(  # doses are found on pro.medicin.dk
            100000,
            200000,
            300000,
            400000,
            500000,
        ),
    )


def olanzapine(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N05AH03",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def clozapine(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N05AH02",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def anxiolytics(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N05B",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=True,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def benzodiazepines(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N05BA",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=True,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def benzodiazepine_related_sleeping_agents(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code=["N05CF01", "N05CF02"],
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=True,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def pregabaline(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N03AX16",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=True,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def opioid_dependence(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    """All opioid dependence medications."""
    return load(
        atc_code="N07BC",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=True,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def buprenorphine(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    """Opioid dependence medications with the active ingredient
    buprenorphine."""
    return load(
        atc_code="N07BC01",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def methadone(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    """Opioid dependence medications with the active ingredient methadone."""
    return load(
        atc_code="N07BC02",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def naxolone(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    """Opioid dependence medications with the active ingredients naxolone and
    buprenorphine."""
    return load(
        atc_code="N07BC51",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def hypnotics(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N05C",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=True,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def hypnotics_and_rivotril(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code=[
            "N05CD08",
            "N05CF01",
            "N05CF02",
            "N05CH01",
            "N05CD02",
            "N05CD05",
            "N05CD06",
            "N05CA01",
            "N05CM18",
            "N05CC01",
            "N05CD03",
            "N05CA04",
            "N03AE01",
        ],
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def mood_stabilisers(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code=["N03AX09", "N03AG01", "N03AF01", "N03AX12"],
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def nervous_system_stimulants(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code=["N06BA04", "N06BA09", "C02AC02"],
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def antidepressives(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N06A",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=True,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


# SSRIs, escitalopram, citalopram, fluvoxamin, fluoxetin, paroxetin


def ssri(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N06AB",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=True,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


# SNRIs, duloxetin, venlafaxin


def snri(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code=["N06AX21", "N06AX16"],
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


# TCAs


def tca(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N06AA",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=True,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def selected_nassa(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code=["N06AX11", "N06AX03"],
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=True,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def lithium(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N05AN01",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def valproate(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N03AG01",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def lamotrigine(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N03AX09",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def hyperactive_disorders_medications(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N06B",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=True,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def dementia_medications(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N06D",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=True,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def anti_epileptics(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N03",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=True,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


# medications used in alcohol abstinence treatment [thiamin, b-combin, klopoxid, fenemal]


def alcohol_abstinence(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code=["A11DA01", "A11EA", "N05BA02", "N03AA02"],
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=False,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )


def analgesic(
    n_rows: int | None = None,
    load_prescribed: bool = False,
    load_administered: bool = True,
    administration_route: str | None = None,
    administration_method: str | None = None,
) -> pd.DataFrame:
    return load(
        atc_code="N02",
        load_prescribed=load_prescribed,
        load_administered=load_administered,
        wildcard_code=True,
        n_rows=n_rows,
        administration_route=administration_route,
        administration_method=administration_method,
    )
