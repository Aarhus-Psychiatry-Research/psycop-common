from pathlib import Path

import pandas as pd

from psycop.projects.clozapine.feature_generation.cohort_definition.clozapine_cohort_definition import (
    ClozapineCohortDefiner,
)
from psycop.projects.clozapine.feature_generation.cohort_definition.eligible_prediction_times.prevalent_serum_clozapine import (
    find_plasma_clozapine_between_2013_2014,
)


def get_latest_version_file_path(file_path: Path) -> Path:
    base_path = file_path.parent
    file_name = file_path.stem
    extension = file_path.suffix

    version = 1
    latest_path = None
    while (candidate := base_path / f"{file_name}_v{version}{extension}").exists():
        latest_path = candidate
        version += 1

    if latest_path is None:
        raise FileNotFoundError(f"No versioned file found for {file_path}")

    return latest_path


def get_all_version_file_paths(file_path: Path) -> list[Path]:
    base_path = file_path.parent
    file_name = file_path.stem
    extension = file_path.suffix

    version = 1
    all_paths = []
    while (candidate := base_path / f"{file_name}_v{version}{extension}").exists():
        all_paths.append(candidate)
        version += 1

    if not all_paths:
        raise FileNotFoundError(f"No versioned files found for {file_path}")

    return all_paths


def load_all_versions(file_path: Path, drop_duplicates: bool = True) -> pd.DataFrame:
    all_paths = get_all_version_file_paths(file_path)
    print(f"Found {len(all_paths)} versions: {[p.name for p in all_paths]}")

    dfs = [pd.read_parquet(p) for p in all_paths]
    combined_df = pd.concat(dfs, ignore_index=True)

    if drop_duplicates:
        before = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=["dw_ek_borger", "timestamp"])
        after = len(combined_df)
        print(f"Dropped {before - after} duplicate rows across versions.")

    return combined_df


def report_counts(df: pd.DataFrame, name: str) -> None:
    total_count = df["dw_ek_borger"].nunique()

    after_2014 = df[df["timestamp"] > "2014-12-31"]
    after_2014_count = after_2014["dw_ek_borger"].nunique()

    between_2014_2016 = df[(df["timestamp"] > "2014-12-31") & (df["timestamp"] <= "2016-09-30")]
    between_2014_2016_count = between_2014_2016["dw_ek_borger"].nunique()

    print(f"\n--- {name} ---")
    print(f"Total: {total_count}")
    print(f"After 2014: {after_2014_count}")
    print(f"Between 2014 and 2016-09-30: {between_2014_2016_count}")


def report_final_cohort_overlap(
    df: pd.DataFrame, name: str, final_cohort_df: pd.DataFrame, outcome_df: pd.DataFrame
) -> None:
    condition_in_final_cohort = df["dw_ek_borger"].isin(final_cohort_df["dw_ek_borger"])
    df_in_final_cohort = df[condition_in_final_cohort]

    print(
        f"Of the remaining {name} dw_ek_borger, "
        f"{df_in_final_cohort['dw_ek_borger'].nunique()} "
        f"are present in the final cohort's prediction times "
        f"(out of {df['dw_ek_borger'].nunique()} remaining)."
    )

    condition_has_outcome = df["dw_ek_borger"].isin(outcome_df["dw_ek_borger"])
    df_with_outcome = df[condition_has_outcome]

    print(
        f"Of the remaining {name} dw_ek_borger, "
        f"{df_with_outcome['dw_ek_borger'].nunique()} "
        f"have the outcome in the final cohort "
        f"(out of {df['dw_ek_borger'].nunique()} remaining)."
    )


# ----- shared data, loaded once -----
validated_text_outcome_clozapine = pd.read_parquet(
    "E:/shared_resources/clozapine/text_outcome/validated_text_outcome_unsure_corrected.parquet"
)

washout_dw_ek_borger = find_plasma_clozapine_between_2013_2014()

filtered_prediction_time_bundle = ClozapineCohortDefiner.get_filtered_prediction_times_bundle()
final_cohort_df = filtered_prediction_time_bundle.prediction_times.frame.to_pandas()

outcome_timestamps_bundle = ClozapineCohortDefiner.get_outcome_timestamps()
outcome_df = outcome_timestamps_bundle.frame.to_pandas()


# ----- historical_prescription -----
historical_prescription_path = get_latest_version_file_path(
    Path(
        "E:/shared_resources/clozapine/text_outcome/unsure/historical_prescription_clozapine.parquet"
    )
)
print(f"Loading: {historical_prescription_path}")

historical_prescription = pd.read_parquet(historical_prescription_path)

condition_already_validated = historical_prescription["dw_ek_borger"].isin(
    validated_text_outcome_clozapine["dw_ek_borger"]
)

condition_washout = historical_prescription["dw_ek_borger"].isin(
    washout_dw_ek_borger["dw_ek_borger"]
)

remaining_historical_prescription = historical_prescription[
    ~condition_already_validated & ~condition_washout
]

print(
    f"Original historical_prescription count: {historical_prescription['dw_ek_borger'].nunique()}"
)
print(
    f"After removing already-validated dw_ek_borger: "
    f"{historical_prescription[~condition_already_validated]['dw_ek_borger'].nunique()}"
)

report_final_cohort_overlap(
    remaining_historical_prescription, "historical_prescription", final_cohort_df, outcome_df
)


# ----- false_incidents -----
false_incidents_path = get_latest_version_file_path(
    Path("E:/shared_resources/clozapine/text_outcome/unsure/false_incidents_clozapine.parquet")
)
print(f"Loading: {false_incidents_path}")

false_incidents = pd.read_parquet(false_incidents_path)

report_counts(false_incidents, "false_incidents")

# Only keep rows with a timestamp after 2014
false_incidents_after_2014 = false_incidents[false_incidents["timestamp"] > "2014-12-31"]

condition_false_already_validated = false_incidents_after_2014["dw_ek_borger"].isin(
    validated_text_outcome_clozapine["dw_ek_borger"]
)

condition_false_washout = false_incidents_after_2014["dw_ek_borger"].isin(
    washout_dw_ek_borger["dw_ek_borger"]
)

remaining_false_incidents = false_incidents_after_2014[
    ~condition_false_already_validated & ~condition_false_washout
]

print(
    f"Original false_incidents count (after 2014): "
    f"{false_incidents_after_2014['dw_ek_borger'].nunique()}"
)

report_final_cohort_overlap(
    remaining_false_incidents, "false_incidents", final_cohort_df, outcome_df
)


# ----- note_clustering -----
note_clustering = load_all_versions(
    Path("E:/shared_resources/clozapine/text_outcome/unsure/note_clustering_clozapine.parquet")
)

report_counts(note_clustering, "note_clustering")

# Only keep rows with a timestamp after 2014
note_clustering_after_2014 = note_clustering[note_clustering["timestamp"] > "2014-12-31"]

condition_clustering_already_validated = note_clustering_after_2014["dw_ek_borger"].isin(
    validated_text_outcome_clozapine["dw_ek_borger"]
)

condition_clustering_washout = note_clustering_after_2014["dw_ek_borger"].isin(
    washout_dw_ek_borger["dw_ek_borger"]
)

remaining_note_clustering = note_clustering_after_2014[
    ~condition_clustering_already_validated & ~condition_clustering_washout
]

print(
    f"Original note_clustering count (after 2014): "
    f"{note_clustering_after_2014['dw_ek_borger'].nunique()}"
)

report_final_cohort_overlap(
    remaining_note_clustering, "note_clustering", final_cohort_df, outcome_df
)
