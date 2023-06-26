import pandas as pd

from psycop.projects.scz_bp.feature_generation.outcome_specification.bp_diagnoses import (
    get_first_bp_diagnosis,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.scz_diagnoses import (
    get_first_scz_diagnosis,
)


def get_first_scz_or_bp_diagnosis() -> pd.DataFrame:
    dfs = {"scz": get_first_scz_diagnosis(), "bp": get_first_bp_diagnosis()}
    for df_type in dfs:
        dfs[df_type]["source"] = df_type

    combined = pd.concat(
        [dfs[k] for k in dfs],
        axis=0,
    )

    first_scz_or_bp_indicator = (
        combined.sort_values("timestamp")
        .groupby("dw_ek_borger")
        .first()
        .reset_index(drop=False)
    )
    first_scz_or_bp_indicator["value"] = 1
    return first_scz_or_bp_indicator[["dw_ek_borger", "timestamp", "value", "source"]]


if __name__ == "__main__":
    df = get_first_scz_or_bp_diagnosis()
    pass