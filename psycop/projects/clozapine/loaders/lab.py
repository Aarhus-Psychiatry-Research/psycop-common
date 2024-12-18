"""Loaders for lab results loading."""

from __future__ import annotations

import pandas as pd

from psycop.common.feature_generation.loaders.non_numerical_coercer import (
    multiply_inequalities_in_df,
)
from psycop.common.feature_generation.loaders.raw.sql_load import sql_load


def _load_non_numerical_values_and_coerce_inequalities(
    blood_sample_id: str | list[str],
    n_rows: int | None,
    view: str,
    ineq2mult: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Load non-numerical values for a blood sample.

    Args:
        blood_sample_id (Union[str, list]): The blood_sample_id, typically an NPU code. If a list, concatenates the values. # noqa: DAR102
        n_rows (Optional[int]): Number of rows to return. Defaults to None.
        view (str): The view to load from.
        ineq2mult (dict[str, float]): A dictionary mapping inequalities to a multiplier. Defaults to None.

    Returns:
        pd.DataFrame: A dataframe with the non-numerical values.
    """
    cols = "dw_ek_borger, datotid_sidstesvar, svar"

    if isinstance(blood_sample_id, list):
        npu_codes = ", ".join([f"'{x}'" for x in blood_sample_id])

        npu_where = f"npukode in ({npu_codes})"
    else:
        npu_where = f"npukode = '{blood_sample_id}'"

    sql = f"SELECT {cols} FROM [fct].{view} WHERE datotid_sidstesvar IS NOT NULL AND {npu_where} AND numerisksvar IS NULL AND (left(Svar,1) = '>' OR left(Svar, 1) = '<')"

    df = sql_load(sql, database="USR_PS_FORSK", n_rows=n_rows)

    df = df.rename(columns={"datotid_sidstesvar": "timestamp", "svar": "value"})

    return multiply_inequalities_in_df(df, ineq2mult=ineq2mult)


def _load_numerical_values(
    blood_sample_id: str | list[str], n_rows: int | None, view: str
) -> pd.DataFrame:
    """Load numerical values for a blood sample.

    Args:
        blood_sample_id (str): The blood_sample_id, typically an NPU code.  # noqa: DAR102
        n_rows (Optional[int]): Number of rows to return. Defaults to None.
        view (str): The view to load from.

    Returns:
        pd.DataFrame: A dataframe with the numerical values.
    """

    cols = "dw_ek_borger, datotid_sidstesvar, numerisksvar"

    if isinstance(blood_sample_id, list):
        npu_codes = ", ".join([f"'{x}'" for x in blood_sample_id])

        npu_where = f"npukode in ({npu_codes})"
    else:
        npu_where = f"npukode = '{blood_sample_id}'"

    sql = f"SELECT {cols} FROM [fct].{view} WHERE datotid_sidstesvar IS NOT NULL AND {npu_where} AND numerisksvar IS NOT NULL"
    df = sql_load(sql, database="USR_PS_FORSK", n_rows=n_rows)

    df = df.rename(columns={"datotid_sidstesvar": "timestamp", "numerisksvar": "value"})

    return df


def _load_cancelled(
    blood_sample_id: str | list[str], n_rows: int | None, view: str
) -> pd.DataFrame:
    """Load cancelled samples for a blood sample.

    Args:
        blood_sample_id (str): The blood_sample_id, typically an NPU code.  # noqa: DAR102
        n_rows (Optional[int]): Number of rows to return. Defaults to None.
        view (str): The view to load from.
    Returns:
        pd.DataFrame: A dataframe with the timestamps for cancelled values.
    """
    cols = "dw_ek_borger, datotid_sidstesvar"

    if isinstance(blood_sample_id, list):
        npu_codes = ", ".join([f"'{x}'" for x in blood_sample_id])

        npu_where = f"npukode in ({npu_codes})"
    else:
        npu_where = f"npukode = '{blood_sample_id}'"

    sql = f"SELECT {cols} FROM [fct].{view} WHERE {npu_where} AND datotid_sidstesvar IS NOT NULL AND Svar = 'Aflyst'"

    df = sql_load(sql, database="USR_PS_FORSK", n_rows=n_rows)

    # Create the value column == 1, since all timestamps here are from cancelled blood samples
    df["value"] = 1

    df = df.rename(columns={"datotid_sidstesvar": "timestamp"})

    return df


def _load_all_values(
    blood_sample_id: str | list[str], n_rows: int | None, view: str
) -> pd.DataFrame:
    """Load all samples for a blood sample.

    Args:
        blood_sample_id (str): The blood_sample_id, typically an NPU code.  # noqa: DAR102
        n_rows (Optional[int]): Number of rows to return. Defaults to None.
        view (str): The view to load from.

    Returns:
        pd.DataFrame: A dataframe with all values.
    """
    cols = "dw_ek_borger, datotid_sidstesvar, svar"

    if isinstance(blood_sample_id, list):
        npu_codes = ", ".join([f"'{x}'" for x in blood_sample_id])

        npu_where = f"npukode in ({npu_codes})"
    else:
        npu_where = f"npukode = '{blood_sample_id}'"

    sql = f"SELECT {cols} FROM [fct].{view} WHERE datotid_sidstesvar IS NOT NULL AND {npu_where}"

    df = sql_load(sql, database="USR_PS_FORSK", n_rows=n_rows)

    df = df.rename(columns={"datotid_sidstesvar": "timestamp", "svar": "value"})

    return df


def blood_sample(
    blood_sample_id: str | list[str],
    n_rows: int | None = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    """Load a blood sample.

    Args:
        blood_sample_id (Union[str, list]): The blood_sample_id, typically an NPU code. If a list, concatenates the values. # noqa: DAR102
        n_rows: Number of rows to return. Defaults to None.
        values_to_load (str): Which values to load. Takes either "numerical", "numerical_and_coerce", "cancelled" or "all". Defaults to None, which is coerced to "all".
    Returns:
        pd.DataFrame
    """

    allowed_values_to_load = ["numerical", "numerical_and_coerce", "cancelled", "all", None]

    dfs = []

    if values_to_load not in allowed_values_to_load:
        raise ValueError(
            f"values_to_load must be one of {allowed_values_to_load}, not {values_to_load}"
        )

    fn_dict = {
        "coerce": _load_non_numerical_values_and_coerce_inequalities,
        "numerical": _load_numerical_values,
        "cancelled": _load_cancelled,
        "all": _load_all_values,
    }

    sources_to_load = [k for k in fn_dict if k in values_to_load]

    n_rows_per_fn = int(n_rows / len(sources_to_load)) if n_rows else None

    view = "[Clozapin_blodproever]"
    for k in sources_to_load:
        dfs.append(
            fn_dict[k](  # type: ignore
                blood_sample_id=blood_sample_id, n_rows=n_rows_per_fn, view=view
            )
        )

    # Concatenate dfs
    df = pd.concat(dfs) if len(dfs) > 1 else dfs[0]

    return df.reset_index(drop=True).drop_duplicates(  # type: ignore
        subset=["dw_ek_borger", "timestamp", "value"], keep="first"
    )


def hba1c(n_rows: int | None = None, values_to_load: str = "numerical_and_coerce") -> pd.DataFrame:
    return blood_sample(
        blood_sample_id=["NPU27300", "AAB00093"], n_rows=n_rows, values_to_load=values_to_load
    )


def triglycerides(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU04094", n_rows=n_rows, values_to_load=values_to_load)


def hdl(n_rows: int | None = None, values_to_load: str = "numerical_and_coerce") -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU01567", n_rows=n_rows, values_to_load=values_to_load)


def ldl(n_rows: int | None = None, values_to_load: str = "numerical_and_coerce") -> pd.DataFrame:
    return blood_sample(
        blood_sample_id=["NPU01568", "AAB00101"], n_rows=n_rows, values_to_load=values_to_load
    )


def alat(n_rows: int | None = None, values_to_load: str = "numerical_and_coerce") -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU19651", n_rows=n_rows, values_to_load=values_to_load)


def asat(n_rows: int | None = None, values_to_load: str = "numerical_and_coerce") -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU19654", n_rows=n_rows, values_to_load=values_to_load)


def alkaline_phosfatase(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id=["NPU57047", "NPU27783"], n_rows=n_rows, values_to_load=values_to_load
    )


def haemoglobin(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU02319", n_rows=n_rows, values_to_load=values_to_load)


def lymphocytes(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU02636", n_rows=n_rows, values_to_load=values_to_load)


def leukocytes(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU02593", n_rows=n_rows, values_to_load=values_to_load)


def neutrophils(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU02902", n_rows=n_rows, values_to_load=values_to_load)


def crp(n_rows: int | None = None, values_to_load: str = "numerical_and_coerce") -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU19748", n_rows=n_rows, values_to_load=values_to_load)


def creatinine(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id=["NPU18016", "ASS00355", "ASS00354"],
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


def egfr(n_rows: int | None = None, values_to_load: str = "numerical_and_coerce") -> pd.DataFrame:
    return blood_sample(
        blood_sample_id=["DNK35302", "DNK35131", "AAB00345", "AAB00343"],
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


def albumine_creatinine_ratio(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU19661", n_rows=n_rows, values_to_load=values_to_load)


def natrium(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU03429", n_rows=n_rows, values_to_load=values_to_load)


def kalium(n_rows: int | None = None, values_to_load: str = "numerical_and_coerce") -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU03230", n_rows=n_rows, values_to_load=values_to_load)


def calcium(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU01443", n_rows=n_rows, values_to_load=values_to_load)


def thrombocytes(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU03568", n_rows=n_rows, values_to_load=values_to_load)


def vitamin_d(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU10267", n_rows=n_rows, values_to_load=values_to_load)


def tsh(n_rows: int | None = None, values_to_load: str = "numerical_and_coerce") -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU03577", n_rows=n_rows, values_to_load=values_to_load)


def vitamin_b12(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU01700", n_rows=n_rows, values_to_load=values_to_load)


def cyp21a2(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU19053", n_rows=n_rows, values_to_load=values_to_load)


def cyp2c19(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU19309", n_rows=n_rows, values_to_load=values_to_load)


def cyp2c9(n_rows: int | None = None, values_to_load: str = "numerical_and_coerce") -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU32095", n_rows=n_rows, values_to_load=values_to_load)


def cyp3a5(n_rows: int | None = None, values_to_load: str = "numerical_and_coerce") -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU27992", n_rows=n_rows, values_to_load=values_to_load)


def cyp2d6(n_rows: int | None = None, values_to_load: str = "numerical_and_coerce") -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU19308", n_rows=n_rows, values_to_load=values_to_load)


def cyp3a4(n_rows: int | None = None, values_to_load: str = "numerical_and_coerce") -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU29776", n_rows=n_rows, values_to_load=values_to_load)


def p_lithium(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU02613", n_rows=n_rows, values_to_load=values_to_load)


def p_clozapine(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU04114", n_rows=n_rows, values_to_load=values_to_load)


def p_olanzapine(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU09358", n_rows=n_rows, values_to_load=values_to_load)


def p_aripiprazol(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU26669", n_rows=n_rows, values_to_load=values_to_load)


def p_risperidone(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU04868", n_rows=n_rows, values_to_load=values_to_load)


def p_paliperidone(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU18359", n_rows=n_rows, values_to_load=values_to_load)


def p_haloperidol(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU03937", n_rows=n_rows, values_to_load=values_to_load)


def p_amitriptyline(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU01224", n_rows=n_rows, values_to_load=values_to_load)


def p_nortriptyline(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU02923", n_rows=n_rows, values_to_load=values_to_load)


def p_clomipramine(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU01616", n_rows=n_rows, values_to_load=values_to_load)


def p_paracetamol(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU03024", n_rows=n_rows, values_to_load=values_to_load)


def p_ibuprofen(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU08794", n_rows=n_rows, values_to_load=values_to_load)


def p_ethanol(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU01992", n_rows=n_rows, values_to_load=values_to_load)
