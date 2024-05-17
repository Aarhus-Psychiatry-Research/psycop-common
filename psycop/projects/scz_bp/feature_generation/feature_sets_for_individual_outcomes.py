# Load metadata for table 1 with column for individual outcomes

# load main feature set and add the cols

# generate synth data for the individual outcomes
import polars as pl
import polars.selectors as cs


from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR

if __name__ == "__main__":
    flattened_datasets_dir = OVARTACI_SHARED_DIR / "scz_bp" / "flattened_datasets"

    metadata_df = pl.read_parquet(
        flattened_datasets_dir / "metadata_for_table1" / "metadata_for_table1.parquet"
    ).sort("pred_time_uuid")
    feature_df = pl.read_parquet(
        flattened_datasets_dir
        / "l1_l4-lookbehind_183_365_730-all_relevant_tfidf_1000_lookbehind_730.parquet"
    ).sort("prediction_time_uuid")


    metadata_df = metadata_df.rename(
        {
            "meta_scz_diagnosis_within_0_to_1825_days_max_fallback_0": "outc_scz_within_1825_days_fallback_0",
            "meta_bp_diagnosis_within_0_to_1825_days_max_fallback_0": "outc_bp_within_1825_days_fallback_0",
        }
    )
    # drop previous outcome cols - add for individual outcomes
    feature_df = feature_df.drop(cs.starts_with("outc_")).with_columns(
        metadata_df["outc_scz_within_1825_days_fallback_0"], metadata_df["outc_bp_within_1825_days_fallback_0"]
    )

    feature_df.write_parquet(flattened_datasets_dir / "l1_l4-lookbehind_183_365_730-all_relevant_tfidf_1000_lookbehind_730_individual_outcomes.parquet")

