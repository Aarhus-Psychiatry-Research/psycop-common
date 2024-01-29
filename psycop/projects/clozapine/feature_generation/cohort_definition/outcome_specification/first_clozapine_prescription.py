import pandas as pd

from psycop.common.feature_generation.loaders.raw.load_medications import clozapine


def get_first_clozapine_prescription() -> pd.DataFrame:
    df = clozapine(load_prescribed=True, load_administered=False)

    df_first_clozapine_presciption = (
        df.sort_values("timestamp", ascending=True)
        .groupby("dw_ek_borger")
        .first()
        .reset_index(drop=False)
    )
    return df_first_clozapine_presciption[["dw_ek_borger", "timestamp", "value"]]
