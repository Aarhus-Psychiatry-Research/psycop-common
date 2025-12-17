from pathlib import Path

import pandas as pd


def load_and_concat_feature_sets(base_path: str | Path, new_name: str) -> pd.DataFrame:
    """
    Load all parquet files from a base path,
    concatenate them vertically, save the merged parquet file,
    and return the merged DataFrame.
    """
    base_path = Path(base_path)

    df_list = []

    # load all parquet files in the base path
    for file_path in base_path.glob("*.parquet"):
        df = pd.read_parquet(file_path)
        df_list.append(df)

    # Stack/concat vertically
    merged_df = pd.concat(df_list, axis=0, ignore_index=True)

    # Save final merged df
    output_path = base_path / f"{new_name}.parquet"
    merged_df.to_parquet(output_path)

    return merged_df


def load_and_join_feature_sets(
    dir_path: str | Path,
    left_file: str,
    right_file: str,
    join_keys: list[str],
    new_name: str,
    how: str = "inner",
) -> pd.DataFrame:
    """
    Load two parquet feature sets from the same directory and join them
    on the given keys
    """

    dir_path = Path(dir_path)

    left_path = dir_path / left_file
    right_path = dir_path / right_file

    # Load data
    left_df = pd.read_parquet(left_path)
    right_df = pd.read_parquet(right_path)

    if len(left_df) != len(right_df):
        raise ValueError(f"Row count mismatch: {len(left_df)} vs {len(right_df)}")

    for name, df in [("left", left_df), ("right", right_df)]:
        if df.duplicated(join_keys).any():
            raise ValueError(f"Duplicate join keys in {name} dataframe")

    merged_df = pd.merge(
        left_df,
        right_df,
        on=join_keys,
        how=how,  # type: ignore
        validate="one_to_one",
        suffixes=("", "_right"),
    )

    if len(merged_df) != len(left_df):
        raise ValueError("Join changed row count — join keys do not match 1-to-1")

    output_path = dir_path / f"{new_name}.parquet"
    merged_df.to_parquet(output_path)

    return merged_df


if __name__ == "__main__":
    FEATURE_SET_DIR = Path("E:/shared_resources/forced_admissions_outpatient/flattened_datasets")

    join_keys = ["dw_ek_borger", "timestamp", "age", "pred_age_in_years"]

    merged = load_and_join_feature_sets(
        dir_path=FEATURE_SET_DIR,
        left_file="structured_features/structured_features.parquet",
        right_file="text_features/text_features.parquet",
        join_keys=join_keys,
        new_name="full_feature_set",
    )
