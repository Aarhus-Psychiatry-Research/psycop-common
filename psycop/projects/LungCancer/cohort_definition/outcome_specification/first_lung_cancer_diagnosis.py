import pandas as pd

from psycop.common.feature_generation.loaders.raw.load_lung_cancer_outcomes import lung_cancer


def get_first_lung_cancer_diagnosis() -> pd.DataFrame:
    df = lung_cancer()

    df_first_lung_cancer_diagnosis = (
        df.sort_values("timestamp", ascending=True)
        .groupby("dw_ek_borger")
        .first()
        .reset_index(drop=False)
    )
    return df_first_lung_cancer_diagnosis[["dw_ek_borger", "timestamp", "value"]]
