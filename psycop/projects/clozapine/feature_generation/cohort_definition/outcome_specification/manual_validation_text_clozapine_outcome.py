from pathlib import Path

import pandas as pd

from psycop.common.feature_generation.utils import write_df_to_file
from psycop.common.global_utils.sql.loader import sql_load


def process_df_from_disk():  # noqa: ANN201
    view = "[raw_text_df_clozapine_outcome]"
    sql = "SELECT * FROM [fct]." + view

    raw_text_df = sql_load(sql, chunksize=None)

    rows_no_text_on_matched_word = pd.read_parquet(
        "E:/shared_resources/clozapine/text_outcome/rows_no_text_on_matched_word_v1.parquet"
    )

    validated_text_outcome_clozapine = pd.read_parquet(
        "E:/shared_resources/clozapine/text_outcome/validated_text_outcome_clozapine_v157.parquet"
    )

    unsure_text_outcome_clozapine = pd.read_parquet(
        "E:/shared_resources/clozapine/text_outcome/unsure_text_outcome_clozapine_v144.parquet"
    )

    no_text_outcome_df = pd.read_parquet(
        "E:/shared_resources/clozapine/text_outcome/no_text_outcome_clozapine_v136.parquet"
    )

    # remove rows from already checked cpr/rows -
    #  Condition for 'rows_no_text_on_matched_word' - specific rows
    condition_no_text_matched_words = (
        raw_text_df[["dw_ek_borger", "timestamp"]]
        .isin(rows_no_text_on_matched_word[["dw_ek_borger", "timestamp"]].to_dict(orient="list"))
        .all(axis=1)
    )

    condition_no_text = (
        raw_text_df[["dw_ek_borger", "timestamp"]]
        .isin(no_text_outcome_df[["dw_ek_borger", "timestamp"]].to_dict(orient="list"))
        .all(axis=1)
    )

    # Condition for 'unsure_text_df' - specific rows
    condition_unsure = (
        raw_text_df[["dw_ek_borger", "timestamp"]]
        .isin(unsure_text_outcome_clozapine[["dw_ek_borger", "timestamp"]].to_dict(orient="list"))
        .all(axis=1)
    )

    # condition for "validated text outcome clozapine"- remove all rows for specific dw_ek_borger
    condition_validated_text = (
        raw_text_df[["dw_ek_borger"]]
        .isin(validated_text_outcome_clozapine[["dw_ek_borger"]].to_dict(orient="list"))
        .all(axis=1)
    )

    # Apply the conditions to filter rows in raw_text_df
    raw_text_df = raw_text_df[
        ~condition_no_text_matched_words
        & ~condition_no_text
        & ~condition_unsure
        & ~condition_validated_text
    ]

    # sort and filter for timestamps more than 2013 raw_text_df for  reading
    raw_text_df = raw_text_df[raw_text_df["timestamp"].dt.year > 2012]

    sorted_df = raw_text_df.sort_values(by=["dw_ek_borger", "timestamp"])

    unique_dw_ek_borger_count = sorted_df["dw_ek_borger"].nunique()
    print(f"Number of unique dw_ek_borger left in sorted_df: {unique_dw_ek_borger_count}")

    return (
        sorted_df,
        validated_text_outcome_clozapine,
        unsure_text_outcome_clozapine,
        no_text_outcome_df,
    )


def get_next_version_file_path(file_path: Path) -> Path:
    base_path = file_path.parent
    file_name = file_path.stem
    extension = file_path.suffix

    version = 1
    while (new_file_path := base_path / f"{file_name}_v{version}{extension}").exists():
        version += 1

    return new_file_path


def read_and_validate_text_for_clozapine_outcome(  # noqa: ANN201
    sorted_df: pd.DataFrame,
    validated_text_outcome_clozapine: pd.DataFrame,
    unsure_text_outcome_clozapine: pd.DataFrame,
    no_text_outcome_df: pd.DataFrame,
):
    no_text_version_count = 0
    unsure_version_count = 0
    validated_version_count = 0

    # Iterate through sorted DataFrame
    for index, row in sorted_df.iterrows():
        if row["dw_ek_borger"] in validated_text_outcome_clozapine["dw_ek_borger"].values:
            print(
                f"Skipping row {index + 1} as dw_ek_borger {row['dw_ek_borger']} is already in validated_text_outcome_df."  # type: ignore
            )
            continue

        # Display the current row information
        print(
            f"\nRow {index + 1} - Fuzz Ratio: {row['fuzz_ratio']}, Matched Word: {row['matched_word']}, dw_ek_borger: {row['dw_ek_borger']}, timestamp: {row['timestamp']}, value: {row['value']}"  # type: ignore
        )

        # Ask the user whether to add to validated_text_outcome_df, unsure, or move to the next row
        choice = input(
            "Do you want to add this text to validated_text_outcome_df? (y/n/unsure/skiprow): "
        ).lower()

        if choice == "y":
            # Append the row to validated_text_outcome_df
            validated_text_outcome_clozapine = validated_text_outcome_clozapine.append(
                row, ignore_index=True
            )  # type: ignore
            print("Added to validated_text_outcome_df.")

            validated_version_count += 1
            if validated_version_count % 5 == 0:
                file_path = Path(
                    "E:/shared_resources/clozapine/text_outcome/validated_text_outcome_clozapine.parquet"
                )
                new_file_path = get_next_version_file_path(file_path)

                columns_to_save = ["dw_ek_borger", "timestamp"]
                df_to_save_val = validated_text_outcome_clozapine[columns_to_save]
                write_df_to_file(df=df_to_save_val, file_path=new_file_path)

                print("saved a new version of validated_text_outcome_clozapine to disk.")

        elif choice == "n":
            # Add to no_text_outcome_df_unique
            no_text_outcome_df = no_text_outcome_df.append(row, ignore_index=True)  # type: ignore
            print("Added to no_text_outcome_df.")

            no_text_version_count += 1
            if no_text_version_count % 20 == 0:
                file_path = Path(
                    "E:/shared_resources/clozapine/text_outcome/no_text_outcome_clozapine.parquet"
                )
                new_file_path = get_next_version_file_path(file_path)

                columns_to_save = ["dw_ek_borger", "timestamp"]
                df_to_save_no_text = no_text_outcome_df[columns_to_save]
                write_df_to_file(df=df_to_save_no_text, file_path=new_file_path)

                print("saved a new version of no_text_outcome_clozapine to disk.")

        elif choice == "unsure":
            # Add to unsure_text_outcome_clozapine
            unsure_text_outcome_clozapine = unsure_text_outcome_clozapine.append(
                row, ignore_index=True
            )  # type: ignore
            print("Added to unsure_text_outcome_clozapine.")

            unsure_version_count += 1
            if unsure_version_count % 5 == 0:
                file_path = Path(
                    "E:/shared_resources/clozapine/text_outcome/unsure_text_outcome_clozapine.parquet"
                )
                columns_to_save = ["dw_ek_borger", "timestamp"]
                df_to_save_unsure = unsure_text_outcome_clozapine[columns_to_save]
                write_df_to_file(df=df_to_save_unsure, file_path=new_file_path)

                print("saved a new version of unsured_text_outcome_clozapine to disk.")

        elif choice == "skip":
            # Skip to the next row
            print("Skipped to the next row.")

    return (
        sorted_df,
        validated_text_outcome_clozapine,
        unsure_text_outcome_clozapine,
        no_text_outcome_df,
    )


if __name__ == "__main__":
    (
        sorted_df,
        validated_text_outcome_clozapine,
        unsure_text_outcome_clozapine,
        no_text_outcome_df,
    ) = process_df_from_disk()

    (
        sorted_df,
        validated_text_outcome_clozapine,
        unsure_text_outcome_clozapine,
        no_text_outcome_df,
    ) = read_and_validate_text_for_clozapine_outcome(
        sorted_df,
        validated_text_outcome_clozapine,
        unsure_text_outcome_clozapine,
        no_text_outcome_df,
    )
