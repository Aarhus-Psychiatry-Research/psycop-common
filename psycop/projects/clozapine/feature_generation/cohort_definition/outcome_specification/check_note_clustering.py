import re
from pathlib import Path
from typing import Optional

import pandas as pd

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.common.feature_generation.utils import write_df_to_file

SAVE_DIR = Path("E:/shared_resources/clozapine/text_outcome/unsure")


def load_all_processed_borgers(
    base_name: str = "note_clustering", parent_dir: Path = SAVE_DIR
) -> set[int]:
    """Automatically finds ALL version files for base_name and collects all unique dw_ek_borger IDs across all of them."""
    if not parent_dir.exists():
        print(f"Directory {parent_dir} does not exist. Starting fresh.")
        return set()

    combined_borgers: set[int] = set()
    found_files = []

    # Match any parquet file that starts with base_name and contains _v<number>
    pattern = rf"^{re.escape(base_name)}.*_v\d+\.parquet$"

    for file_path in parent_dir.glob("*.parquet"):
        if re.search(pattern, file_path.name, re.IGNORECASE):
            found_files.append(file_path)

    if not found_files:
        print(f"No previous version files found for '{base_name}'. Starting fresh.")
        return set()

    print(f"Found {len(found_files)} previous version file(s):")
    for file_path in sorted(found_files):
        try:
            df_saved = pd.read_parquet(file_path)
            if "dw_ek_borger" in df_saved.columns:
                file_borgers = set(df_saved["dw_ek_borger"].dropna().unique())
                combined_borgers.update(file_borgers)
                print(f"  - {file_path.name}: {len(file_borgers)} patient(s)")
        except Exception as e:
            print(f"  - Failed to read {file_path.name}: {e}")

    print(f"Total unique patient IDs collected across all versions: {len(combined_borgers)}")
    return combined_borgers


def get_next_version_file_path(file_path: Path) -> Path:
    base_path = file_path.parent
    file_name = file_path.stem
    extension = file_path.suffix

    version = 1
    while (new_file_path := base_path / f"{file_name}_v{version}{extension}").exists():
        version += 1

    return new_file_path


def _save_group(df: pd.DataFrame, name: str, count: int, every: int = 5) -> None:
    if count % every == 0:
        clean_name = name if name.endswith("clozapine") else f"{name}_clozapine"
        file_path = SAVE_DIR / f"{clean_name}.parquet"

        new_file_path = get_next_version_file_path(file_path)
        columns_to_save = ["dw_ek_borger", "timestamp"]
        write_df_to_file(df=df[columns_to_save], file_path=new_file_path)
        print(f"Saved a new version to disk: {new_file_path.name}")


def get_note_clustering_for_reading(
    min_notes: int = 3, dw_ek_borger: Optional[int] = None
) -> pd.DataFrame:
    view = "[raw_text_df_clozapine_outcome]"
    sql = "SELECT * FROM [fct]." + view

    raw_text_df = sql_load(sql)

    unsure_text_outcome_clozapine = pd.read_parquet(
        "E:/shared_resources/clozapine/text_outcome/unsure_text_outcome_clozapine_v97.parquet"
    )

    # Narrow raw_text_df down to only the dw_ek_borger present in unsure_text_outcome_clozapine
    raw_text_df = raw_text_df[
        raw_text_df["dw_ek_borger"].isin(unsure_text_outcome_clozapine["dw_ek_borger"])
    ]

    # Automatically load and combine dw_ek_borger from ALL saved version files
    already_processed_borgers = load_all_processed_borgers(
        base_name="note_clustering", parent_dir=SAVE_DIR
    )

    if already_processed_borgers:
        initial_count = raw_text_df["dw_ek_borger"].nunique()
        raw_text_df = raw_text_df[~raw_text_df["dw_ek_borger"].isin(already_processed_borgers)]
        remaining_count = raw_text_df["dw_ek_borger"].nunique()
        print(
            f"Excluded {initial_count - remaining_count} already processed patient(s). "
            f"Remaining to process: {remaining_count}"
        )

    # Count notes per dw_ek_borger + exact timestamp (date + time)
    notes_per_timestamp = (
        raw_text_df.groupby(["dw_ek_borger", "timestamp"]).size().reset_index(name="note_count")
    )

    # Keep only dw_ek_borger + timestamp combos with min_notes or more (e.g., >= 3)
    clustered = notes_per_timestamp[notes_per_timestamp["note_count"] >= min_notes]

    # Filter raw_text_df down to only rows belonging to these exact clustered timestamps
    merged_df = raw_text_df.merge(
        clustered[["dw_ek_borger", "timestamp"]], on=["dw_ek_borger", "timestamp"], how="inner"
    )

    sorted_df = merged_df.sort_values(by=["dw_ek_borger", "timestamp"])

    # Filter for a single patient if explicitly passed
    if dw_ek_borger is not None:
        sorted_df = sorted_df[sorted_df["dw_ek_borger"] == dw_ek_borger]

    return sorted_df


def sort_note_clustering_interactive(df: pd.DataFrame) -> pd.DataFrame:
    """Interactively reviews clustered notes and saves confirmed matches to parquet.

    Controls:
      - 'y' / 'yes' : Save row to note_clustering, increment count/save, move to NEXT patient
      - 'n' / 'no'  : Do NOT save row, move to NEXT patient
      - [Enter]     : Move to NEXT row for current patient
      - 'q' / 'quit': Stop and exit reading session
    """
    note_clustering = pd.DataFrame(columns=df.columns)
    clustering_count = 0

    groups = list(df.groupby("dw_ek_borger", sort=False))
    total_patients = len(groups)

    if total_patients == 0:
        print("No remaining patients to process.")
        return note_clustering

    print(f"\nFound {total_patients} unique dw_ek_borger(s) left to review.\n")

    for idx, (dw_ek_borger, group) in enumerate(groups, start=1):
        remaining = total_patients - idx
        print(
            f"\n==================== dw_ek_borger: {dw_ek_borger} "
            f"(Patient {idx} of {total_patients} | {remaining} left) ===================="
        )

        skip_patient = False
        for _, row in group.iterrows():
            print(
                f"  Fuzz Ratio: {row['fuzz_ratio']}, Matched Word: {row['matched_word']}, "
                f"timestamp: {row['timestamp']}, value: {row['value']}"  # type: ignore
            )

            user_input = (
                input(
                    "--> [y] Save & Next Patient | [n] Skip Patient | [Enter] Next Row | [q] Quit: "
                )
                .strip()
                .lower()
            )

            if user_input in ("y", "yes"):
                note_clustering = pd.concat([note_clustering, row.to_frame().T], ignore_index=True)
                clustering_count += 1
                print(f"Added to note_clustering (Session saved: {clustering_count}).")
                _save_group(note_clustering, "note_clustering", clustering_count)

                skip_patient = True
                break  # Jump to next dw_ek_borger

            if user_input in ("n", "no"):
                print(f"Skipping remaining notes for dw_ek_borger {dw_ek_borger}...")
                skip_patient = True
                break  # Jump to next dw_ek_borger

            if user_input in ("q", "quit"):
                print("Exiting interactive reader...")
                return note_clustering

        if skip_patient:
            continue

    return note_clustering


if __name__ == "__main__":
    sorted_df = get_note_clustering_for_reading(min_notes=10)

    note_clustering_df = sort_note_clustering_interactive(sorted_df)
