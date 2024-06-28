import polars as pl

from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR, TEXT_EMBEDDINGS_DIR


def load_bp_feature_set(feature_set_name: str) -> pl.DataFrame:
    feature_set_dir = (
        OVARTACI_SHARED_DIR
        / "bipolar"
        / "flattened_datasets"
        / feature_set_name
        / f"{feature_set_name}.parquet"
    )
    return pl.read_parquet(feature_set_dir)


def load_embedded_text(name: str) -> pl.DataFrame:
    return pl.read_parquet(TEXT_EMBEDDINGS_DIR / name)


if __name__ == "__main__":
    df = load_bp_feature_set("structured_predictors_2_layer_interval_days_100")

    