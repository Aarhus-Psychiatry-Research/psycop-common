import pandas as pd

from psycop.projects.ect.feature_generation.cohort_definition.outcome_specification.procedure_codes import (
    get_ect_procedures,
)


def get_first_ect_indicator() -> pd.DataFrame:
    
    procedure_codes = get_ect_procedures().rename({"procedurekodetekst": "cause"})
    first_ect = procedure_codes.sort("timestamp").groupby("dw_ek_borger").first()

    return first_ect.select(["dw_ek_borger", "timestamp", "cause"]).to_pandas()


if __name__ == "__main__":
    df = get_first_ect_indicator()
