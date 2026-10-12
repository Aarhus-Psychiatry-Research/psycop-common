import pandas as pd

from psycop.common.feature_generation.loaders_2025.diagnoses import type_2_diabetes


def get_first_type_2_diabetes_diagnosis() -> pd.DataFrame:
    df = type_2_diabetes()

    df_first_t2d_diag = (
        df.sort_values("timestamp").groupby("dw_ek_borger").first().reset_index(drop=False)
    )

    return df_first_t2d_diag[["dw_ek_borger", "timestamp"]]


if __name__ == "__main__":
    df = get_first_type_2_diabetes_diagnosis()
