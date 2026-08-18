### This script merges the old and new flattened data for secondary analysis with modified sliding window temporal validation ###

"""
This script merges the original and new flattened data for secondary analysis with modified sliding window temporal validation.
Reports:
  - new patients (dw_ek_borger) introduced by the "newest" dataset
  - patients present in both datasets
  - earliest and latest timestamp across the merged data

"""

from pathlib import Path

import pandas as pd

# ---------------- CONFIG ----------------
BASE_DIR = Path("E:/shared_resources/")

ORIGINAL_DATASET = (
    BASE_DIR
    / "forced_admissions_inpatient"
    / "flattened_datasets"
    / "structured_feature_set"
    / "structured_feature_set.parquet"
)
TEMPORAL_VAL_DATASET = (
    BASE_DIR
    / "forced_admissions_inpatient_temp_val"
    / "flattened_datasets"
    / "structured_feature_set_temp_val.parquet"
)

TIMESTAMP_COL = "timestamp"
PATIENT_COL = "dw_ek_borger"

OUTPUT_FILE = (
    BASE_DIR
    / "forced_admissions_inpatient_temp_val"
    / "flattened_datasets"
    / "merged_original_temp_val_feature_set.parquet"
)
# -----------------------------------------


def merge_datasets():
    df_original = pd.read_parquet(ORIGINAL_DATASET)
    df_temp_val = pd.read_parquet(TEMPORAL_VAL_DATASET)

    # pred_sex_female is stored as bool in the df_original but int (0/1) in the
    # df_temp_val one; convert the old dataset's column to int so both sides match.
    df_original["pred_sex_female"] = df_original["pred_sex_female"].astype(int)

    print(
        f"Old dataset: {ORIGINAL_DATASET.name} -> {len(df_original):,} rows, {df_original[PATIENT_COL].nunique():,} unique patients"
    )
    print(
        f"New dataset: {TEMPORAL_VAL_DATASET.name} -> {len(df_temp_val):,} rows, {df_temp_val[PATIENT_COL].nunique():,} unique patients"
    )

    old_patients = set(df_original[PATIENT_COL].unique())
    new_patients = set(df_temp_val[PATIENT_COL].unique())

    added_patients = new_patients - old_patients
    overlapping_patients = new_patients & old_patients

    # ---- Per-dataset timestamp boundaries (before merging) ----
    ts_original = pd.to_datetime(df_original[TIMESTAMP_COL])
    ts_temp_val = pd.to_datetime(df_temp_val[TIMESTAMP_COL])

    latest_in_original = ts_original.max()
    earliest_in_temp_val = ts_temp_val.min()

    # ---- Merge: keep all rows from both ----
    merged = pd.concat([df_original, df_temp_val], ignore_index=True)

    # Optional: drop exact duplicate rows if the same row exists in both files
    before_dedup = len(merged)
    merged = merged.drop_duplicates()
    after_dedup = len(merged)

    # ---- Timestamps across merged data ----
    ts = pd.to_datetime(merged[TIMESTAMP_COL])
    earliest = ts.min()
    latest = ts.max()

    print("\n--- MERGE REPORT ---")
    print(f"Total rows after merge (before dedup): {before_dedup:,}")
    if before_dedup != after_dedup:
        print(
            f"Total rows after merge (after dropping {before_dedup - after_dedup:,} exact duplicate rows): {after_dedup:,}"
        )
    print(f"Total unique patients in merged data: {merged[PATIENT_COL].nunique():,}")
    print(
        f"New patients added from newest dataset ({TEMPORAL_VAL_DATASET.name}): {len(added_patients):,}"
    )
    print(f"Patients present in BOTH datasets: {len(overlapping_patients):,}")
    print(f"Earliest timestamp (merged): {earliest}")
    print(f"Latest timestamp (merged): {latest}")
    print(f"Latest timestamp in old dataset ({ORIGINAL_DATASET.name}): {latest_in_original}")
    print(
        f"Earliest timestamp in temporal validation dataset ({TEMPORAL_VAL_DATASET.name}): {earliest_in_temp_val}"
    )

    # ---- Save merged result ----
    merged.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nMerged file saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    merge_datasets()
