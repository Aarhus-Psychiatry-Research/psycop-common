import polars as pl

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load

ECT_PROCEDURE_CODES = {
    "Coercion": [
        "BRTB1 - Tvangsbehandling med elektrokonvulsiv terapi (ECT)",
        "BRTB10 - Tvangsbehandling med ECT unilateralt",
        "BRTB10A - Tvangsbehandling med ECT unilateralt pga. helbred",
        "BRTB11 - Tvangsbehandling med ECT bilateralt",
        "BRTB11A - Tvangsbehandling med ECT bilateralt pga. helbred",
        "BRTB11B - Tvangsbehandling med ECT bilateralt pga. farlighed",
    ],
    "Non-coercion": [
        "BRXA1 - Behandling med elektrokonvulsiv terapi (ECT)",
        "BRXA10 - Behandling med unilateral elektrokonvulsiv terapi (ECT)",
        "BRXA11 - Behandling med bilateral elektrokonvulsiv terapi (ECT)",
    ],
}


def get_ect_procedures(coercion_filter: str | None = None) -> pl.DataFrame:
    """
    Retrieve ECT procedures with an optional coercion filter.

    Args:
        coercion_filter (str): Filter to include 'coercion', 'non-coercion', or None for all. Default is None.

    Returns:
        pl.DataFrame: A dataframe of the selected procedures.
    """
    cols = "[dw_ek_borger], [datotid_udfoert], [procedurekodetekst], procedureart"
    table = "[USR_PS_Forsk].[fct].[Clozapin_ECT]"

    # Filter ECT_PROCEDURE_CODES based on the coercion_filter argument
    if coercion_filter == "Coercion":
        procedure_codes = ECT_PROCEDURE_CODES.get("Coercion", [])
    elif coercion_filter == "Non-coercion":
        procedure_codes = ECT_PROCEDURE_CODES.get("Non-coercion", [])
    else:
        # Include all if no filter is specified
        procedure_codes = [item for sublist in ECT_PROCEDURE_CODES.values() for item in sublist]

    ect_procedure_codes_str = f"""'{"', '".join(procedure_codes)}'"""

    df = (
        pl.from_pandas(
            sql_load(
                query=f"SELECT {cols} FROM {table} WHERE procedurekodetekst IN ({ect_procedure_codes_str}) AND procedureart = 'P'"
            )
        )
        .rename({"datotid_udfoert": "timestamp"})
        .drop("procedureart")
    )
    return df


def ect_coercion() -> pl.DataFrame:
    return get_ect_procedures(coercion_filter="Coercion")


def ect_non_coercion() -> pl.DataFrame:
    return get_ect_procedures(coercion_filter="Non-coercion")


def ect_all() -> pl.DataFrame:
    return get_ect_procedures(coercion_filter=None)


if __name__ == "__main__":
    df = ect_all()
