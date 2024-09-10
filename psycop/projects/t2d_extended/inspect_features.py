import polars

if __name__ == "__main__":
    df = polars.read_parquet(
        "E:/shared_resources/t2d_extended/flattened_datasets/t2d_extended_feature_set/t2d_extended_feature_set.parquet"
    )
    pass
