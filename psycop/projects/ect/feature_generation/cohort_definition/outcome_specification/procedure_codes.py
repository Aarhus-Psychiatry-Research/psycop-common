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
    "Not coercion": [
        "BRXA1 - Behandling med elektrokonvulsiv terapi (ECT)",
        "BRXA10 - Behandling med unilateral elektrokonvulsiv terapi (ECT)",
        "BRXA11 - Behandling med bilateral elektrokonvulsiv terapi (ECT)",
    ],
}


def get_ect_procedures() -> pl.DataFrame:
    cols = "[dw_ek_borger], [datotid_udfoert], [procedurekodetekst], procedureart"
    table = (
        "[USR_PS_Forsk].[fct].[FOR_alle_procedurekoder_uanset_art_psyk_somatik_inkl_2021_feb2022]"
    )

    ect_procedure_code_list = [item for sublist in ECT_PROCEDURE_CODES.values() for item in sublist]
    ect_procedure_codes_str = f"""'{"', '".join(ect_procedure_code_list)}'"""

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
