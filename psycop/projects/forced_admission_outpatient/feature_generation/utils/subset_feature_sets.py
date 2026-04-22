"""Script for subsetting special feature sets from full feature set"""

from pathlib import Path

import pandas as pd

FEATURE_SET_DIR = Path("E:/shared_resources/forced_admissions_outpatient/flattened_datasets")


def subset_feature_df(feature_set_path: str):
    splits = ["train", "val", "test"]

    FULL_PATH = FEATURE_SET_DIR / feature_set_path  # type: ignore

    for split in splits:
        full_df = pd.read_parquet(FULL_PATH / f"{split}.parquet")

        # Get column subsets
        sent_trans_cols = [col for col in full_df.columns if col.startswith("pred_pred_sent")]  # type: ignore
        tfidf_cols = list(
            filter(
                lambda x: x not in set(sent_trans_cols),
                [col for col in full_df.columns if col.startswith("pred_pred_")],
            )
        )
        all_pred_cols = [col for col in full_df.columns if col.startswith("pred_")]  # type: ignore
        non_text_pred_cols = list(set(all_pred_cols) - set(tfidf_cols) - set(sent_trans_cols))
        non_text_pred_cols.remove("pred_age_in_years")
        non_text_pred_cols.remove("pred_sex_female")

        # Only tfidf
        tfidf_df = full_df.copy().drop(columns=sent_trans_cols + non_text_pred_cols)

        tfidf_path = FULL_PATH.parent / "tfidf_750_feature_set"

        Path.mkdir(tfidf_path, exist_ok=True)

        tfidf_df.to_parquet(tfidf_path / f"{split}.parquet")

        # Only sentence embeddings
        sent_trans_df = full_df.copy().drop(columns=tfidf_cols + non_text_pred_cols)

        sent_trans_path = FULL_PATH.parent / "sent_trans_feature_set"

        Path.mkdir(sent_trans_path, exist_ok=True)

        sent_trans_df.to_parquet(sent_trans_path / f"{split}.parquet")

        # Structured + tfidf
        structured_and_tfidf_df = full_df.copy().drop(columns=sent_trans_cols)

        structured_and_tfidf_path = FULL_PATH.parent / "structured_and_tfidf_750_feature_set"

        Path.mkdir(structured_and_tfidf_path, exist_ok=True)

        structured_and_tfidf_df.to_parquet(structured_and_tfidf_path / f"{split}.parquet")

        # Structured + sentence embeddings
        structured_and_sent_trans_df = full_df.copy().drop(columns=tfidf_cols)

        structured_and_sent_trans_path = FULL_PATH.parent / "structured_and_sent_trans_feature_set"

        Path.mkdir(structured_and_sent_trans_path, exist_ok=True)

        structured_and_sent_trans_df.to_parquet(structured_and_sent_trans_path / f"{split}.parquet")

        # Only structured
        structured_df = full_df.copy().drop(columns=tfidf_cols + sent_trans_cols)

        structured_path = FULL_PATH.parent / "structured_feature_set"

        Path.mkdir(structured_path, exist_ok=True)

        structured_df.to_parquet(structured_path / f"{split}.parquet")


def subset_text_features(feature_set_path: str, output_path: str):
    FULL_FEATURE_SET_PATH = FEATURE_SET_DIR / feature_set_path
    FULL_OUTPUT_PATH = FEATURE_SET_DIR / output_path

    # Load feature set
    df = pd.read_parquet(FULL_FEATURE_SET_PATH)

    # Prefixes to keep
    keep_prefixes = ("pred_pred_tfidf", "pred_pred_sen_emb", "pred_sex_female", "pred_age_in_years")

    # Identify columns to drop
    cols_to_drop = [
        col for col in df.columns if col.startswith("pred_") and not col.startswith(keep_prefixes)
    ]

    # Subset dataframe
    df = df.drop(columns=cols_to_drop)

    # Ensure output directory exists
    Path.mkdir(FULL_OUTPUT_PATH, exist_ok=True)

    # Save result
    df.to_parquet(FULL_OUTPUT_PATH / "text_feature_set.parquet")


if __name__ == "__main__":
    subset_text_features("full_feature_set", "text_feature_set")
