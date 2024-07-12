import pandas as pd

from psycop.projects.clozapine.feature_generation.cohort_definition.outcome_specification.first_clozapine_prescription import (
    get_first_clozapine_prescription,
)


def combine_structured_and_text_outcome():  # noqa: ANN201
    validated_text_outcome_clozapine = pd.read_parquet(
        "E:/shared_resources/clozapine/text_outcome/validated_text_outcome_unsure_corrected.parquet"
    )
    validated_text_outcome_clozapine["value"] = 1

    structured_outcomes = get_first_clozapine_prescription()

    structured_outcomes = structured_outcomes[
        structured_outcomes["timestamp"] > pd.Timestamp("2016-10-01")
    ]

    # Merge the dataframes
    combined_df = pd.concat([validated_text_outcome_clozapine, structured_outcomes])

    # For each dw_ek_borger, if there is a timestamp in text outcome before 1 October 2016, keep that
    # Otherwise, keep the earliest timestamp from structured outcome
    combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])
    earliest_outcome_df = combined_df.sort_values(by=["dw_ek_borger", "timestamp"]).drop_duplicates(
        subset="dw_ek_borger", keep="first"
    )
    post_october_2016_structured = structured_outcomes[["dw_ek_borger", "timestamp"]].set_index(
        "dw_ek_borger"
    )

    def select_timestamp(row: pd.Series) -> pd.Series:
        dw_ek_borger = row["dw_ek_borger"]
        earliest_timestamp = row["timestamp"]
        if dw_ek_borger in post_october_2016_structured.index:
            structured_row = post_october_2016_structured.loc[dw_ek_borger]
            structured_timestamp = structured_row["timestamp"]
            if earliest_timestamp >= pd.Timestamp("2016-10-01"):
                return pd.Series(
                    [structured_timestamp, structured_row["value"]], index=["timestamp", "value"]
                )
        return pd.Series([earliest_timestamp, pd.NA], index=["timestamp", "value"])

    combined_clozapine_outcome = earliest_outcome_df.set_index("dw_ek_borger")
    combined_clozapine_outcome["timestamp"] = combined_clozapine_outcome.apply(
        select_timestamp,
        axis=1,  # noqa ANN202
    )
    combined_clozapine_outcome.reset_index(inplace=True)  # noqa: PD002

    unique_dw_ek_borger_final = combined_clozapine_outcome["dw_ek_borger"].nunique()
    print(
        f"Unique dw_ek_borger in the combined structured and text outcome: {unique_dw_ek_borger_final}"
    )

    return combined_clozapine_outcome


if __name__ == "__main__":
    combine_structured_and_text_outcome()
