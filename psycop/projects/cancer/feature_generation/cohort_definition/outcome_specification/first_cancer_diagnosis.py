import pandas as pd

from psycop.common.feature_generation.loaders.raw.load_cancer_outcomes import any_cancer


def get_first_cancer_diagnosis() -> pd.DataFrame:
    df = any_cancer()

    df_first_cancer_diagnosis = (
        df.sort_values("timestamp", ascending=True)
        .groupby("dw_ek_borger")
        .first()
        .reset_index(drop=False)
    )
    return df_first_cancer_diagnosis[["dw_ek_borger", "timestamp", "value"]]
