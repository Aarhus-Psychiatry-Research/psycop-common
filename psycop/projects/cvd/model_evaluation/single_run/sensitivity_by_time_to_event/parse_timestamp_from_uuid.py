import polars as pl


def parse_timestamp_from_uuid(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("pred_time_uuid")
        .str.split("-")
        .list.slice(1)
        .list.join("-")
        .str.strptime(pl.Datetime, format="%Y-%m-%d-%H-%M-%S")
        .alias("timestamp")
    )
