from pathlib import Path
import pandas as pd


def load_and_merge_feature_sets(base_path: str | Path, new_name: str) -> pd.DataFrame:
    """
    Load and merge train, validation, and test feature sets from the specified base path.

    Parameters:
        base_path (str | Path): Directory containing train/validation/test subfolders.
        new_name (str): Base name to use when saving merged parquet files.

    Returns:
        dict[str, pd.DataFrame]: Merged DataFrames for each set.
    """
    base_path = Path(base_path)
    feature_sets = ["train", "validation", "test"]

    for feature_set in feature_sets:
        feature_dir = base_path / feature_set

        # collect parquet files
        parquet_files = sorted(feature_dir.glob("*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {feature_dir}")

        # load and merge
        df_list = [pd.read_parquet(p) for p in parquet_files]
        merged_df = pd.concat(df_list, axis=1)

        # save merged file
        output_path = base_path / f"{feature_set}.parquet"
        merged_df.to_parquet(output_path)

    return merged_df  # type: ignore


if __name__ == "__main__":
    FEATURE_SET_DIR = Path(
        "E:/shared_resources/forced_admissions_inpatient/flattened_datasets/structured_feature_set"
    )

    merged = load_and_merge_feature_sets(FEATURE_SET_DIR, "structured_feature_set")
