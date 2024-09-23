import pandas as pd


def correct_unsure_cases():  # noqa: ANN201
    validated_text_outcome_clozapine = pd.read_parquet(
        "E:/shared_resources/clozapine/text_outcome/validated_text_outcome_clozapine_v129.parquet"
    )

    unsure_validated_text_outcome_clozapine = pd.read_parquet(
        "E:/shared_resources/clozapine/text_outcome/unsure/unsure_validated_text_outcome_clozapine_v50.parquet"
    )

    prevalent_cases = pd.read_parquet(
        "E:/shared_resources/clozapine/text_outcome/unsure/prevalent_cases_clozapine_v41.parquet"
    )

    non_prevalent_cases = pd.read_parquet(
        "E:/shared_resources/clozapine/text_outcome/unsure/non_prevalent_cases_clozapine_v124.parquet"
    )

    # Remove dw_ek_borger in non_prevalent_cases from validated_text_outcome_clozapine
    validated_text_outcome_clozapine = validated_text_outcome_clozapine[
        ~validated_text_outcome_clozapine["dw_ek_borger"].isin(non_prevalent_cases["dw_ek_borger"])
    ]

    # Update timestamp to '2013-01-01 00:00:00' for dw_ek_borger in prevalent_cases
    validated_text_outcome_clozapine.loc[
        validated_text_outcome_clozapine["dw_ek_borger"].isin(prevalent_cases["dw_ek_borger"]),
        "timestamp",
    ] = pd.Timestamp("2013-01-01 00:00:00")

    # Add dw_ek_borger and timestamp from unsure_validated_text_outcome_clozapine if not already in validated_text_outcome_clozapine
    unsure_to_add = unsure_validated_text_outcome_clozapine[
        ~unsure_validated_text_outcome_clozapine["dw_ek_borger"].isin(
            validated_text_outcome_clozapine["dw_ek_borger"]
        )
    ]

    # Create the new dataframe
    validated_text_outcome_unsure_corrected = pd.concat(
        [validated_text_outcome_clozapine, unsure_to_add]
    )

    # Sort by dw_ek_borger and timestamp and drop duplicates, keeping the earliest timestamp
    validated_text_outcome_unsure_corrected = validated_text_outcome_unsure_corrected.sort_values(
        by=["dw_ek_borger", "timestamp"]
    ).drop_duplicates(subset="dw_ek_borger", keep="first")

    # Get the number of unique dw_ek_borger in both dataframes
    unique_dw_ek_borger_validated = validated_text_outcome_clozapine["dw_ek_borger"].nunique()
    unique_dw_ek_borger_corrected = validated_text_outcome_unsure_corrected[
        "dw_ek_borger"
    ].nunique()

    # Print the unique counts
    print(f"Unique dw_ek_borger in validated text outcome: {unique_dw_ek_borger_validated}")
    print(
        f"Unique dw_ek_borger in validated text outcome unsure corrected: {unique_dw_ek_borger_corrected}"
    )

    # Save the new dataframe to disk
    validated_text_outcome_unsure_corrected.to_parquet(
        "E:/shared_resources/clozapine/text_outcome/validated_text_outcome_unsure_corrected.parquet"
    )


if __name__ == "__main__":
    correct_unsure_cases()
