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

    view = "[CVD_T2D_Labka]"
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


def unscheduled_p_glc(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    npu_suffixes = ["02192", "21531"]

    blood_sample_ids = [f"NPU{suffix}" for suffix in npu_suffixes]

    return blood_sample(
        blood_sample_id=blood_sample_ids, n_rows=n_rows, values_to_load=values_to_load
    )


def ogtt(n_rows: int | None = None, values_to_load: str = "numerical_and_coerce") -> pd.DataFrame:
    return blood_sample(blood_sample_id="NPU04177", n_rows=n_rows, values_to_load=values_to_load)


def fasting_p_glc(
    n_rows: int | None = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(blood_sample_id="DNK35842", n_rows=n_rows, values_to_load=values_to_load)
