import polars as pl
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR


def load_scz_bp_feature_set(feature_set_name: str):
    dir = (
        OVARTACI_SHARED_DIR
        / "scz_bp"
        / "initial_feature_set"
        / "flattened_datasets"
        / feature_set_name
    )
    return pl.read_parquet(dir / "train.parquet")


if __name__ == "__main__":
    df = load_scz_bp_feature_set("layer1")
