from collections.abc import Collection
from pathlib import Path

import polars as pl

from psycop.common.feature_generation.loaders.raw.load_visits import physical_visits


def load_shak_to_location_mapping() -> pl.DataFrame:
    """vest = herning, holstebro, gødstrup
    midt = silkeborg, viborg, skive
    øst = horsens, aarhus, risskov, randers"""
    # shak mapping from https://sor-filer.sundhedsdata.dk/sor_produktion/data/shak/shakcomplete/shakcomplete.txt
    return (
        pl.read_csv(
            Path(__file__).parent / "shak_mapping.csv",
        )
        .with_columns(
            pl.col("shak_6").cast(str),
        )
        .rename({"grouping": "region"})
    )


def non_adult_psychiatry_shak() -> list[str]:
    """Dropping Børne og ungdomspsykiatrisk afdeling/center as we don't use
    data from child psychiatry. Dropping central visitation as it is just a
    place of administration"""
    return ["660011", "660020", "660021"]


def get_first_visit_at_each_region_by_patient(df: pl.DataFrame) -> pl.DataFrame:
    return df.groupby(["dw_ek_borger", "region"], maintain_order=True).first()


def get_first_visit_by_patient(df: pl.DataFrame) -> pl.DataFrame:
    return df.groupby(
        "dw_ek_borger",
    ).first()


def get_first_visit_at_second_region_by_patient(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.groupby("dw_ek_borger")
        .apply(lambda group: group[1])
        .rename(
            {"timestamp": "first_regional_move_timestamp", "region": "second_region"},
        )
    )


def add_migration_date_by_patient(
    first_visit_at_first_region_df: pl.DataFrame,
    first_visit_at_second_region_df: pl.DataFrame,
) -> pl.DataFrame:
    """Add the timestamp of the first visit at a different region to the
    first_visit_at_first_region_df. Set this timestamp to
    2100-01-01 if the patient has only received treatment in one region."""
    return (
        first_visit_at_first_region_df.join(
            first_visit_at_second_region_df,
            how="left",
            on="dw_ek_borger",
        ).drop("timestamp")
    ).with_columns(
        pl.when(pl.col("first_regional_move_timestamp").is_null())
        .then(
            "2100-01-01 00:00:00",
        )  # set cutoff to 2100 if patient only has visits in one region
        .otherwise(pl.col("first_regional_move_timestamp"))
        .str.strptime(pl.Datetime)
        .alias("first_regional_move_timestamp"),
    )


def add_shak_to_region_mapping(
    visits: pl.DataFrame,
    shak_to_location_df: pl.DataFrame,
    shak_codes_to_drop: Collection[str],
) -> pl.DataFrame:
    """Add the region to each visit based on the shak code of the visit. Drops
    visits at shak codes specified in shak_codes_to_drop."""
    return (
        (
            visits.with_columns(
                pl.col("shak_location").str.slice(offset=0, length=6).alias("shak_6"),
            ).join(shak_to_location_df, on="shak_6", how="left")
        )
        .filter(~pl.col("shak_6").is_in(shak_codes_to_drop))
        .select("dw_ek_borger", "timestamp", "region")
    )


def get_regional_split_df() -> pl.LazyFrame:
    """Return a dataframe with the region at which each patient first had
    a contact. If a patient had contacts at multiple regions, the timestamp
    of the first contact at a different region is also included."""
    shak_to_location_df = load_shak_to_location_mapping()

    visits = pl.from_pandas(physical_visits(shak_code=6600, return_shak_location=True))

    sorted_all_visits_df = add_shak_to_region_mapping(
        visits=visits,
        shak_to_location_df=shak_to_location_df,
        shak_codes_to_drop=non_adult_psychiatry_shak(),
    ).sort(["dw_ek_borger", "timestamp"])

    # find timestamp of first visit at each different region
    first_visit_at_each_region = get_first_visit_at_each_region_by_patient(
        df=sorted_all_visits_df,
    )

    # get the first visit
    first_visit_at_first_region = get_first_visit_by_patient(
        df=first_visit_at_each_region,
    )

    # get the first visit at a different region
    first_visit_at_second_region = get_first_visit_at_second_region_by_patient(
        df=first_visit_at_each_region,
    )

    # add the migration date for each patient
    geographical_split_df = add_migration_date_by_patient(
        first_visit_at_first_region,
        first_visit_at_second_region,
    )
    # add indicator for which split each patient belongs to
    geographical_split_df = geographical_split_df.with_columns(
        pl.when(pl.col("region") == "øst")
        .then("train")
        .when(pl.col("region") == "vest")
        .then("val")
        .otherwise("test")
        .alias("split"),
    )

    return geographical_split_df.select(
        "dw_ek_borger",
        "region",
        "first_regional_move_timestamp",
        "split",
    ).lazy()


if __name__ == "__main__":
    get_regional_split_df()