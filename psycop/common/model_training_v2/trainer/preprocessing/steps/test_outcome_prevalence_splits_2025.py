import pandas as pd

from psycop.common.feature_generation.loaders.raw.load_visits import (
    admissions,
    physical_visits_to_psychiatry,
)
from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.projects.cancer.feature_generation.cohort_definition.outcome_specification.first_cancer_diagnosis import (
    get_first_cancer_diagnosis,
)
from psycop.projects.cvd.feature_generation.cohort_definition.outcome_specification.combined import (
    get_first_cvd_indicator,
)
from psycop.projects.ect.feature_generation.cohort_definition.outcome_specification.combined import (
    get_first_ect_indicator,
)
from psycop.projects.forced_admission_inpatient.cohort.extract_admissions_and_visits.get_forced_admissions import (
    forced_admissions_onset_timestamps,
)
from psycop.projects.forced_admission_inpatient.cohort.extract_admissions_and_visits.get_inpatient_admissions_to_psychiatry import (
    admissions_discharge_timestamps,
)
from psycop.projects.restraint.feature_generation.modules.loaders.load_restraint_outcome_timestamps import (
    load_restraint_outcome_timestamps,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.bp_diagnoses import (
    get_first_bp_diagnosis,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.scz_diagnoses import (
    get_first_scz_diagnosis,
)
from psycop.projects.t2d.feature_generation.cohort_definition.outcome_specification.lab_results import (
    get_first_diabetes_lab_result_above_threshold,
)


def load_split_ids_2025(split: str) -> pd.DataFrame:
    view = f"[psycop_{split}_ids_2025]"
    sql = f"SELECT * FROM [fct].{view}"
    df = sql_load(sql)
    return df


if __name__ == "__main__":
    splits = ["train", "val", "test"]

    # === Load prediction-time cohorts ===
    inv_admission_cohort_df = admissions_discharge_timestamps()
    cancer_cohort_df = physical_visits_to_psychiatry()
    cvd_cohort_df = physical_visits_to_psychiatry()
    t2d_cohort_df = physical_visits_to_psychiatry()
    scz_cohort_df = physical_visits_to_psychiatry()
    ect_cohort_df = admissions()
    restraint_cohort_df = admissions()

    # === Load outcome data ===
    inv_admission_outcome_df = forced_admissions_onset_timestamps()
    cancer_outcome_df = get_first_cancer_diagnosis()
    cvd_outcome_df = get_first_cvd_indicator()
    t2d_outcome_df = get_first_diabetes_lab_result_above_threshold()
    scz_outcome_df = get_first_scz_diagnosis()
    bp_outcome_df = get_first_bp_diagnosis()
    ect_outcome_df = get_first_ect_indicator()
    restraint_outcome_df = load_restraint_outcome_timestamps().rename(
        columns={"datotid_start": "timestamp"}
    )

    # === Pair cohort with outcome ===
    outcome_cohort_pairs = {
        "inv_admission": (inv_admission_cohort_df, inv_admission_outcome_df),
        "cancer": (cancer_cohort_df, cancer_outcome_df),
        "CVD": (cvd_cohort_df, cvd_outcome_df),
        "T2D": (t2d_cohort_df, t2d_outcome_df),
        "scz": (scz_cohort_df, scz_outcome_df),
        "BP": (scz_cohort_df, bp_outcome_df),
        "ECT": (ect_cohort_df, ect_outcome_df),
        "restraint": (restraint_cohort_df, restraint_outcome_df),
    }

    summary_all = []

    for outcome_name, (cohort_df, outcome_df) in outcome_cohort_pairs.items():
        print(f"\n=== PROCESSING OUTCOME: {outcome_name} ===")

        cohort_df_2013 = cohort_df[cohort_df["timestamp"] >= "2013-01-01"]

        # Clean & deduplicate cohort and outcome
        cohort_df_unique = cohort_df_2013.drop_duplicates(
            subset=["dw_ek_borger"], keep="first"
        ).drop(columns=["timestamp"], errors="ignore")
        outcome_unique = outcome_df.drop_duplicates(subset=["dw_ek_borger"], keep="first").drop(
            columns=["timestamp"], errors="ignore"
        )

        cohort_ids = set(cohort_df_unique["dw_ek_borger"])
        print(f"Cohort size (unique patients) for {outcome_name}: {len(cohort_ids)}")

        # We'll collect cohort IDs that are assigned to any split (i.e. cohort ∩ split)
        all_cohort_ids_in_splits = pd.Series(dtype="int64")

        summary = []
        split_outcome_counts = {}

        for split in splits:
            split_ids = load_split_ids_2025(split)

            # IMPORTANT: restrict to cohort first — this ensures we only count cohort patients that fall in the split
            filtered_df = split_ids.merge(cohort_df_unique, on="dw_ek_borger", how="inner")

            # collect the cohort ids that are present in this split
            all_cohort_ids_in_splits = pd.concat(
                [all_cohort_ids_in_splits, filtered_df["dw_ek_borger"]]
            )

            # compute totals and outcome presence using the filtered cohort subset
            total = len(filtered_df)
            count_1 = (filtered_df["dw_ek_borger"].isin(outcome_unique["dw_ek_borger"])).sum()
            count_0 = total - count_1
            pct_with_outcome = count_1 / total if total > 0 else 0

            split_outcome_counts[split] = count_1

            print(f"--- {split.upper()} ---")
            print(f"Cohort patients in split: {total}")
            print(f"Outcome present among those: {count_1} ({pct_with_outcome:.2%})")

            summary.append(
                {
                    "outcome": outcome_name,
                    "split": split,
                    "total_patients_in_split": total,
                    "outcome_patients_in_split": count_1,
                    "proportion_with_outcome": round(pct_with_outcome, 4),
                }
            )

        # --- proportion of outcomes across splits (unchanged logic) ---
        total_outcomes_all_splits = sum(split_outcome_counts.values())
        for entry in summary:
            split = entry["split"]
            entry["prop_of_total_outcomes"] = (
                split_outcome_counts[split] / total_outcomes_all_splits
                if total_outcomes_all_splits > 0
                else 0
            )

        # --- Coverage check: compare cohort IDs to cohort IDs *seen in splits* ---
        all_cohort_ids_in_splits_unique = set(all_cohort_ids_in_splits.drop_duplicates())
        missing_from_splits = cohort_ids - all_cohort_ids_in_splits_unique

        if missing_from_splits:
            missing_count = len(missing_from_splits)
            missing_pct = missing_count / len(cohort_ids) * 100 if len(cohort_ids) > 0 else 0.0
            print(
                f"⚠️  {missing_count} cohort patients ({missing_pct:.3f}%) for {outcome_name} are NOT in any split (train/val/test)."
            )
        else:
            print(
                f"All {len(cohort_ids)} cohort patients for {outcome_name} are represented in the splits."
            )

        summary_all.append(pd.DataFrame(summary))

    # === Combine all outcomes into one long table ===
    final_summary = pd.concat(summary_all, ignore_index=True)
    print("\n=== FULL SUMMARY TABLE (PER OUTCOME & SPLIT) ===")
    print(final_summary)

    # === Create compact pivoted summary table ===
    # First compute total patient proportions across splits
    compact_base = final_summary[["outcome", "split", "prop_of_total_outcomes"]]

    compact_summary = (
        compact_base.pivot(index="outcome", columns="split", values="prop_of_total_outcomes")
        .reindex(columns=["train", "val", "test"])  # ensure proper order
        .fillna(0)
        .reset_index()
    ).round(4)

    print("\n=== COMPACT SUMMARY TABLE (proportion of all outcomes) ===")
    print(compact_summary)
