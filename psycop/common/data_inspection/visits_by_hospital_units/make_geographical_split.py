"""Script to get the ids for a split based on geography. Saves a .parquet file
with columns: dw_ek_borger, region, second_region, cutoff_timestamp.

cutoff_timestamp indicates the first time a patient moves to a different region
for treatment (7570 cases) and can be used to drop visits after this date. Is null
if the patient has only received treatment in one region.
"""

from pathlib import Path

import polars as pl

from psycop.common.feature_generation.loaders.raw.load_visits import physical_visits

GEOGRAPHICAL_SPLIT_PATH = Path("E:/shared_resources/splits/geographical_split.parquet")


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
        (
            pl.from_pandas(visits)
            .with_columns(
                pl.col("shak_location").str.slice(offset=0, length=6).alias("shak_6"),
            )
            .join(shak_to_location_df, on="shak_6", how="left")
        )
        .filter(~pl.col("shak_6").is_in(shak_codes_to_drop()))
        .select("dw_ek_borger", "timestamp", "grouping")
    )

    # find timestamp of first visit at each different region
    first_visit_at_each_region = (
        df.sort(["dw_ek_borger", "timestamp"])
        .groupby(["dw_ek_borger", "grouping"], maintain_order=True)
        .first()
    )

    # get the first visit
    first_visit_at_first_region = first_visit_at_each_region.groupby(
        "dw_ek_borger",
    ).first()

    # get the first visit at a different region
    first_visit_at_second_region = (
        first_visit_at_each_region.groupby("dw_ek_borger")
        .apply(lambda group: group[1])
        .rename({"timestamp": "cutoff_timestamp", "grouping": "second_region"})
    )

    geographical_split_df = (
        first_visit_at_first_region.join(
            first_visit_at_second_region,
            how="left",
            on="dw_ek_borger",
        )
        .rename(
            {
                "grouping": "region",
            },
        )
        .drop("timestamp")
    ).with_columns(
        pl.when(pl.col("cutoff_timestamp").is_null())
        .then(
            "2100-01-01 00:00:00",
        )  # set cutoff to 2100 if patient only has visits in one region
        .otherwise(pl.col("cutoff_timestamp"))
        .str.strptime(pl.Datetime),
    )

    GEOGRAPHICAL_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    geographical_split_df.write_parquet(file=GEOGRAPHICAL_SPLIT_PATH)
