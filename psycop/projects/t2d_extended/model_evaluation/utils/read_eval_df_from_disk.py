import polars as pl

def read_eval_df_from_disk(experiment_path: str) -> pl.DataFrame:
    return pl.read_parquet(experiment_path + "/eval_df.parquet")