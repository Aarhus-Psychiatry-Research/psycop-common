import pandas as pd

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.projects.clozapine.feature_generation.cohort_definition.eligible_prediction_times.schizophrenia_diagnosis import (
    add_only_patients_with_schizo,
)
from psycop.projects.clozapine.feature_generation.cohort_definition.outcome_specification.combine_text_structured_clozapine_outcome import (
    combine_structured_and_text_outcome,
)


def load_split_ids_2025(split: str) -> pd.DataFrame:
    view = f"[psycop_{split}_ids_2025]"
    sql = f"SELECT * FROM [fct].{view}"
    df = sql_load(sql)
    return df


if __name__ == "__main__":
    splits = ["train", "val", "test"]

    # Load cohort and outcome data
    cohort_df = add_only_patients_with_schizo().to_pandas()
    outcome_df = combine_structured_and_text_outcome()

    cohort_df = cohort_df[cohort_df["timestamp"] >= "2013-01-01"]

    cohort_df = cohort_df.drop_duplicates(subset=["dw_ek_borger"], keep="first").drop(
        columns="timestamp"
    )

    cohort_df["outcome"] = cohort_df["dw_ek_borger"].isin(outcome_df["dw_ek_borger"]).astype(int)

    print(f"Cohort size after cleaning: {len(cohort_df)}")
    print(f"Patients with outcome=1: {(cohort_df['outcome'] == 1).sum()}")
    print(f"Patients with outcome=0: {(cohort_df['outcome'] == 0).sum()}")

    summary = []
    all_split_ids = pd.Series(dtype="int64")  # store all IDs across splits

    for split in splits:
        split_ids = load_split_ids_2025(split)

        # Track all IDs across splits
        all_split_ids = pd.concat([all_split_ids, split_ids["dw_ek_borger"]])

        # Keep only schizophrenia patients and their outcome
        filtered_df = split_ids.merge(cohort_df, on="dw_ek_borger", how="inner")

        # Count totals
        total = len(filtered_df)
        count_0 = (filtered_df["outcome"] == 0).sum()
        count_1 = (filtered_df["outcome"] == 1).sum()
        pct_1 = count_1 / total if total > 0 else 0

        summary.append(
            {
                "split": split,
                "total": total,
                "outcome_0": count_0,
                "outcome_1": count_1,
                "pct_outcome_1": round(pct_1, 3),
            }
        )

        print(f"=== {split.upper()} SPLIT ===")
        print(f"Total patients after filtering: {total}")
        print(f"Outcome 0: {count_0} ({count_0 / total:.2%})")
        print(f"Outcome 1: {count_1} ({count_1 / total:.2%})\n")

    # Check: are all cohort IDs included in at least one split?
    all_split_ids_unique = all_split_ids.drop_duplicates()
    missing_ids = set(cohort_df["dw_ek_borger"]) - set(all_split_ids_unique)

    if missing_ids:
        print(
            f" WARNING: {len(missing_ids)} patients in cohort_df "
            f"are missing from all splits (train/val/test)."
        )
    else:
        print("All cohort patients are represented in at least one split.")

    # Combine results into a summary DataFrame
    summary_df = pd.DataFrame(summary)
    # Calculate proportion of patients in each split
    total_patients_all_splits = summary_df["total"].sum()
    summary_df["prop_of_total_patients"] = round(summary_df["total"] / total_patients_all_splits, 4)
    print("\n=== SUMMARY TABLE ===")
    print(summary_df)
