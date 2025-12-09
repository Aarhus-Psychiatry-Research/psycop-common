import polars as pl

from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR


def load_bp_feature_set(feature_set_name: str) -> pl.DataFrame:
    feature_set_dir = (
        OVARTACI_SHARED_DIR
        / "bipolar"
        / "flattened_datasets"
        / "flattened_datasets"
        / feature_set_name
        / f"{feature_set_name}.parquet"
    )
    return pl.read_parquet(feature_set_dir)


if __name__ == "__main__":
    df = load_bp_feature_set("bipolar_full_feature_set_interval_days_150")
    print("Hi")
