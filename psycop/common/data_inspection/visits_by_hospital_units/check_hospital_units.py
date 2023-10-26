from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from psycop.common.feature_generation.loaders.raw.load_visits import physical_visits


def count_and_sort_values_of_series(series: pl.Series) -> pl.DataFrame:
    return series.value_counts().sort(by="counts", descending=True)

def load_shak_to_location_mapping() -> pl.DataFrame:
    # from https://sor-filer.sundhedsdata.dk/sor_produktion/data/shak/shakcomplete/shakcomplete.txt
    return pl.read_csv(Path(__file__).parent / "shak_mapping.csv").with_columns(pl.col("shak_6").cast(str))



if __name__ == "__main__":
    shak_to_location_df = load_shak_to_location_mapping()
    
    visits = physical_visits(shak_code=6600, return_shak_location=True)

    df = pl.from_pandas(visits).with_columns(
        pl.col("shak_location").str.slice(offset=0, length=6).alias("shak_6")
    ).join(shak_to_location_df, on="shak_6", how="left")

    pl.Config.set_fmt_str_lengths(100) # to print full cell content

    # number of  visits at each location
    count_and_sort_values_of_series(df["department"])
    count_and_sort_values_of_series(df["unit"])


    # number of unique locations (cities) visited by patient
    df.groupby("dw_ek_borger").agg(pl.col("unit").n_unique().alias("n_unique_locations"))["n_unique_locations"].value_counts()






