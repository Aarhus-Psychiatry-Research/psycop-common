"""Script for subsetting special feature sets from full feature set"""
from pathlib import Path

import pandas as pd

FEATURE_SET_DIR = Path(
    "E:/shared_resources/forced_admissions_inpatient/flattened_datasets",
)


def subset_feature_df(feature_set_path: str):
    splits = ["train", "val", "test"]

    FULL_PATH = FEATURE_SET_DIR / feature_set_path  # type: ignore

    for split in splits:
        full_df = pd.read_parquet(FULL_PATH / f"{split}.parquet")

        # Get column subsets
        sent_trans_cols = [col for col in full_df.columns if col.startswith("pred_pred_sent")]  # type: ignore
        tfidf_cols = [col for col in full_df.columns if col.startswith("pred_pred_tfidf")]  # type: ignore
        all_pred_cols = [col for col in full_df.columns if col.startswith("pred_")]  # type: ignore
        non_text_pred_cols = list(
            set(all_pred_cols) - set(tfidf_cols) - set(sent_trans_cols),
        )
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

        structured_and_tfidf_path = (
            FULL_PATH.parent / "structured_and_tfidf_750_feature_set"
        )

        Path.mkdir(structured_and_tfidf_path, exist_ok=True)

        structured_and_tfidf_df.to_parquet(
            structured_and_tfidf_path / f"{split}.parquet",
        )

        # Structured + sentence embeddings
        structured_and_sent_trans_df = full_df.copy().drop(columns=tfidf_cols)

        structured_and_sent_trans_path = (
            FULL_PATH.parent / "structured_and_sent_trans_feature_set"
        )

        Path.mkdir(structured_and_sent_trans_path, exist_ok=True)

        structured_and_sent_trans_df.to_parquet(
            structured_and_sent_trans_path / f"{split}.parquet",
        )

        # Only structured
        structured_df = full_df.copy().drop(columns=tfidf_cols + sent_trans_cols)

        structured_path = FULL_PATH.parent / "structured_feature_set"

        Path.mkdir(structured_path, exist_ok=True)

        structured_df.to_parquet(structured_path / f"{split}.parquet")


if __name__ == "__main__":
    subset_feature_df(
        "no_washout_feature_sets/full_feature_set_with_sentence_transformers_and_tfidf_750_no_washout",
    )
    subset_feature_df("full_feature_set_with_sentence_transformers_and_tfidf_750")
