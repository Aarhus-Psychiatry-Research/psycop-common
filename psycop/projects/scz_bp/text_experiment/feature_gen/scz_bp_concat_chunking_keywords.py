import polars as pl

from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR

if __name__ == "__main__":
    chunked_dir = OVARTACI_SHARED_DIR / "scz_bp" / "text_exp" / "keywords" / "flattened_datasets"

    df_paths = list(chunked_dir.iterdir())

    df = pl.read_parquet(list(df_paths[0].iterdir())[0])

    for path in df_paths[1:]:
        chunk_df = pl.read_parquet(list(path.iterdir())[0]).drop("timestamp", "dw_ek_borger")
        df = df.join(chunk_df, on="pred_time_uuid", validate="1:1")

    df.write_parquet(chunked_dir.parent / "keywords.parquet")
    
