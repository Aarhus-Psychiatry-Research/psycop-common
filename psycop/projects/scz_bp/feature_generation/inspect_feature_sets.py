import polars as pl

from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR, TEXT_EMBEDDINGS_DIR


def load_scz_bp_feature_set(feature_set_name: str) -> pl.DataFrame:
    feature_set_dir = (
        OVARTACI_SHARED_DIR
        / "scz_bp"
        / "initial_feature_set"
        / "flattened_datasets"
        / feature_set_name
    )
    return pl.read_parquet(feature_set_dir / "train.parquet")


def load_embedded_text(name: str) -> pl.DataFrame:
    return pl.read_parquet(TEXT_EMBEDDINGS_DIR / name)


if __name__ == "__main__":
    for p in TEXT_EMBEDDINGS_DIR.glob("text_embeddings_*"):
        df = load_embedded_text(p.name)
        print(f"Feature set: {p} \n\nShape = {df.shape}")
