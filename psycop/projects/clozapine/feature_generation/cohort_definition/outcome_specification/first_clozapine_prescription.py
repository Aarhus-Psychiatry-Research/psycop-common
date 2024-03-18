import pandas as pd

from psycop.common.feature_generation.loaders.raw.load_medications import clozapine


def get_first_clozapine_prescription(
    timestamps_only: bool = False, timestamp_as_value_col: bool = False
) -> pd.DataFrame:
    df = clozapine(load_prescribed=True, load_administered=False)

    df_first_clozapine_presciption = (
        df.sort_values("timestamp", ascending=True)
        .groupby("dw_ek_borger")
        .first()
        .reset_index(drop=False)
    )

    if timestamp_as_value_col:
        df_first_clozapine_presciption["value"] = df_first_clozapine_presciption["timestamp"].copy()
    else:
        df_first_clozapine_presciption["value"] = 1

    if timestamps_only:
        return df_first_clozapine_presciption[["dw_ek_borger", "timestamp"]]

    return df_first_clozapine_presciption
