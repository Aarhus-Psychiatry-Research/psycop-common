from pathlib import Path

import pandas as pd


def load_and_merge_feature_sets(base_path: str | Path, new_name: str) -> pd.DataFrame:
    """
    Load train, validation, and test parquet files from a base path,
    concatenate them vertically, save the merged parquet file,
    and return the merged DataFrame.
    """
    base_path = Path(base_path)
    feature_sets = ["train", "val", "test"]

    df_list = []

    for feature_set in feature_sets:
        file_path = base_path / f"{feature_set}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        df_list.append(pd.read_parquet(file_path))

    # Stack/concat vertically
    merged_df = pd.concat(df_list, axis=0, ignore_index=True)

    # Save final merged df
    output_path = base_path / f"{new_name}.parquet"
    merged_df.to_parquet(output_path)

    return merged_df


if __name__ == "__main__":
    FEATURE_SET_DIR = Path(
        "E:/shared_resources/forced_admissions_outpatient/flattened_datasets/structured_feature_set"
    )

    merged = load_and_merge_feature_sets(FEATURE_SET_DIR, "structured_feature_set")
