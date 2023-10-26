from pathlib import Path

import polars as pl

from psycop.common.feature_generation.loaders.raw.load_visits import physical_visits


def count_and_sort_values_of_series(series: pl.Series) -> pl.DataFrame:
    return series.value_counts().sort(by="counts", descending=True)


def load_shak_to_location_mapping() -> pl.DataFrame:
    """vest = herning, holstebro, gødstrup
    midt = silkeborg, viborg, skive
    øst = horsens, aarhus, risskov, randers"""
    # shak mapping from https://sor-filer.sundhedsdata.dk/sor_produktion/data/shak/shakcomplete/shakcomplete.txt
    return pl.read_csv(Path(__file__).parent / "shak_mapping.csv").with_columns(
        pl.col("shak_6").cast(str),
    )


def shak_codes_to_drop() -> list[str]:
    """Dropping Børne og ungdomspsykiatrisk afdeling/center as we don't use
    data from child psychiatry. Dropping central visitation as it is just a
    place of administration"""
    return ["660011", "660020", "660021"]


if __name__ == "__main__":
    shak_to_location_df = load_shak_to_location_mapping()

    visits = physical_visits(shak_code=6600, return_shak_location=True)

    df = (
        pl.from_pandas(visits)
        .with_columns(
            pl.col("shak_location").str.slice(offset=0, length=6).alias("shak_6"),
        )
        .join(shak_to_location_df, on="shak_6", how="left")
    ).filter(~pl.col("shak_6").is_in(shak_codes_to_drop()))

    pl.Config.set_fmt_str_lengths(100)  # to print full cell content

    # number of  visits at each location
    count_and_sort_values_of_series(df["department"])
    count_and_sort_values_of_series(df["unit"])
    count_and_sort_values_of_series(df["grouping"])

    # number of unique groupings visited by patient
    n_locations_and_visits_by_patient = df.groupby("dw_ek_borger").agg(
        pl.col("grouping").n_unique().alias("n_unique_locations"),
        pl.col("grouping").count().alias("n_visits"),
    )

    n_locations_and_visits_by_patient.groupby("n_unique_locations").agg(
        pl.col("n_visits").count().alias("n_patients"),
        pl.col("n_visits").sum().alias("n_visits"),
    ).sort("n_unique_locations")

    df.groupby("grouping").agg(
        pl.col("dw_ek_borger").unique().count().alias("n_patients"),
        pl.col("dw_ek_borger").count().alias("n_visits"),
    ).sort("n_patients", descending=True)





