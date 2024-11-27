import pandas as pd

from psycop.common.feature_generation.loaders_2024.medications import antidiabetics_2024


def get_first_antidiabetic_medication() -> pd.DataFrame:
    df = antidiabetics_2024()

    # Group by person id and sort by timestamp, then get the first row for each person
    df_first_antidiabetic_medication = (
        df.sort_values("timestamp").groupby("dw_ek_borger").first().reset_index(drop=False)
    )

    return df_first_antidiabetic_medication[["dw_ek_borger", "timestamp"]]


if __name__ == "__main__":
    df = get_first_antidiabetic_medication()
