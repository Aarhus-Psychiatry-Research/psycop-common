from pathlib import Path

import pandas as pd

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.common.feature_generation.utils import write_df_to_file


def get_relevant_dfs():  # noqa: ANN201
    view = "[raw_text_df_clozapine_outcome]"
    sql = "SELECT * FROM [fct]." + view

    raw_text_df = sql_load(sql)

    validated_text_outcome_clozapine = pd.read_parquet(
        "E:/shared_resources/clozapine/text_outcome/validated_text_outcome_clozapine_v129.parquet"
    )

    unsure_text_outcome_clozapine = pd.read_parquet(
        "E:/shared_resources/clozapine/text_outcome/unsure_text_outcome_clozapine_v97.parquet"
    )

    unsure_validated_text_outcome_clozapine = pd.read_parquet(
        "E:/shared_resources/clozapine/text_outcome/unsure/unsure_validated_text_outcome_clozapine_v50.parquet"
    )

    unsure_no_text_outcome_clozapine = pd.read_parquet(
        "E:/shared_resources/clozapine/text_outcome/unsure/unsure_no_text_outcome_clozapine_v48.parquet"
    )

    prevalent_cases = pd.read_parquet(
        "E:/shared_resources/clozapine/text_outcome/unsure/prevalent_cases_clozapine_v41.parquet"
    )

    non_prevalent_cases = pd.read_parquet(
        "E:/shared_resources/clozapine/text_outcome/unsure/non_prevalent_cases_clozapine_v124.parquet"
    )

    # timestamps more than 2013 raw_text_df for reading
    raw_text_df = raw_text_df[
        raw_text_df["dw_ek_borger"].isin(unsure_text_outcome_clozapine["dw_ek_borger"])
    ]
    raw_text_df = raw_text_df[raw_text_df["timestamp"].dt.year > 2012]

    # remove already validated_text_outcome_clozapine (e.g. first added to unsure_text_outcome_clozapine, then added to validated_text_outcome_clozapine)

    condition_validated_text = raw_text_df["dw_ek_borger"].isin(
        validated_text_outcome_clozapine["dw_ek_borger"]
    )

    # filter for already read rows
    condition_non_prevalent_cases = (
        raw_text_df[["dw_ek_borger", "timestamp"]]  # type: ignore
        .isin(non_prevalent_cases[["dw_ek_borger", "timestamp"]].to_dict(orient="list"))
        .all(axis=1)
    )

    condition_no_text = (
        raw_text_df[["dw_ek_borger", "timestamp"]]  # type: ignore
        .isin(
            unsure_no_text_outcome_clozapine[["dw_ek_borger", "timestamp"]].to_dict(orient="list")
        )
        .all(axis=1)
    )

    # condition for "validated text outcome clozapine"- remove all rows for specific dw_ek_borger

    condition_unsure_validated_text = raw_text_df["dw_ek_borger"].isin(
        unsure_validated_text_outcome_clozapine["dw_ek_borger"]
    )

    condition_prevalent_cases = raw_text_df["dw_ek_borger"].isin(prevalent_cases["dw_ek_borger"])

    # Apply the conditions to filter rows in raw_text_df
    raw_text_df = raw_text_df[  # type: ignore
        ~condition_validated_text
        & ~condition_unsure_validated_text
        & ~condition_prevalent_cases
        & ~condition_non_prevalent_cases
        & ~condition_no_text
    ]

    sorted_df = raw_text_df.sort_values(by=["dw_ek_borger", "timestamp"])

    unique_dw_ek_borger_count = sorted_df["dw_ek_borger"].nunique()
    print(f"Number of unique dw_ek_borger left in unsure_sorted_df: {unique_dw_ek_borger_count}")

    return (
        sorted_df,
        unsure_no_text_outcome_clozapine,
        unsure_validated_text_outcome_clozapine,
        prevalent_cases,
        non_prevalent_cases,
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
    unsure_validated_text_outcome_clozapine: pd.DataFrame,
    unsure_no_text_outcome_clozapine: pd.DataFrame,
    prevalent_cases: pd.DataFrame,
    non_prevalent_cases: pd.DataFrame,
):
    no_text_version_count = 0
    validated_version_count = 0
    prevalent_version_count = 0

    # Iterate through sorted DataFrame
    for index, row in sorted_df.iterrows():
        if (
            row["dw_ek_borger"] in unsure_validated_text_outcome_clozapine["dw_ek_borger"].values
        ) or (row["dw_ek_borger"] in prevalent_cases["dw_ek_borger"].values):
            print(
                f"Skipping row {index + 1} as dw_ek_borger {row['dw_ek_borger']} is already in unsure_validated_text_outcome_clozapine or prevalent cases."  # type: ignore
            )
            continue
        # Display the current row information
        print(
            f"\nRow {index + 1} - Fuzz Ratio: {row['fuzz_ratio']}, Matched Word: {row['matched_word']}, dw_ek_borger: {row['dw_ek_borger']}, timestamp: {row['timestamp']}, value: {row['value']}"  # type: ignore
        )

        # Ask the user whether to add to validated_text_outcome_df, unsure, or move to the next row
        choice = input(
            "Do you want to add this text to unsure_validated_text_outcome_df? (y/n/unsure/skiprow): "
        ).lower()

        if choice == "y":
            # Append the row to unsure_validated_text_outcome_df
            unsure_validated_text_outcome_clozapine = (
                unsure_validated_text_outcome_clozapine.append(row, ignore_index=True)
            )  # type: ignore
            print("Added to unsure_validated_text_outcome_df.")

            validated_version_count += 1
            if validated_version_count % 5 == 0:
                file_path = Path(
                    "E:/shared_resources/clozapine/text_outcome/unsure/unsure_validated_text_outcome_clozapine.parquet"
                )
                new_file_path = get_next_version_file_path(file_path)

                columns_to_save = ["dw_ek_borger", "timestamp"]
                df_to_save_val = unsure_validated_text_outcome_clozapine[columns_to_save]
                write_df_to_file(df=df_to_save_val, file_path=new_file_path)

                print("saved a new version of unsure_  validated_text_outcome_clozapine to disk.")

        elif choice == "n":
            # Add to no_text_outcome_df_unique
            unsure_no_text_outcome_clozapine = unsure_no_text_outcome_clozapine.append(
                row, ignore_index=True
            )  # type: ignore
            print("Added to unsure_no_text_outcome_df.")

            no_text_version_count += 1
            if no_text_version_count % 20 == 0:
                file_path = Path(
                    "E:/shared_resources/clozapine/text_outcome/unsure/unsure_no_text_outcome_clozapine.parquet"
                )
                new_file_path = get_next_version_file_path(file_path)

                columns_to_save = ["dw_ek_borger", "timestamp"]
                df_to_save_no_text = unsure_no_text_outcome_clozapine[columns_to_save]
                write_df_to_file(df=df_to_save_no_text, file_path=new_file_path)

                print("saved a new version of unsure_no_text_outcome_clozapine to disk.")

        elif choice == "prevalent":
            # Add to prevalent_cases
            prevalent_cases = prevalent_cases.append(row, ignore_index=True)  # type: ignore
            print("Added to prevalent_cases.")

            prevalent_version_count += 1
            if prevalent_version_count % 5 == 0:
                file_path = Path(
                    "E:/shared_resources/clozapine/text_outcome/unsure/prevalent_cases_clozapine.parquet"
                )
                new_file_path = get_next_version_file_path(file_path)

                columns_to_save = ["dw_ek_borger", "timestamp"]
                df_to_save_prevalent = prevalent_cases[columns_to_save]
                write_df_to_file(df=df_to_save_prevalent, file_path=new_file_path)  # type: ignore

                print("saved a new version of prevalent_cases to disk.")

        elif choice == "non-prevalent":
            # Add to non-prevalent_cases
            non_prevalent_cases = non_prevalent_cases.append(row, ignore_index=True)  # type: ignore
            print("Added to non_prevalent_cases.")

            prevalent_version_count += 1
            if prevalent_version_count % 5 == 0:
                file_path = Path(
                    "E:/shared_resources/clozapine/text_outcome/unsure/non_prevalent_cases_clozapine.parquet"
                )
                new_file_path = get_next_version_file_path(file_path)

                columns_to_save = ["dw_ek_borger", "timestamp"]
                df_to_save_non_prevalent = non_prevalent_cases[columns_to_save]
                write_df_to_file(df=df_to_save_non_prevalent, file_path=new_file_path)  # type: ignore

                print("saved a new version of non_prevalent_cases to disk.")

    return (
        sorted_df,
        unsure_validated_text_outcome_clozapine,
        unsure_no_text_outcome_clozapine,
        prevalent_cases,
        non_prevalent_cases,
    )


if __name__ == "__main__":
    (
        sorted_df,
        unsure_no_text_outcome_clozapine,
        unsure_validated_text_outcome_clozapine,
        prevalent_cases,
        non_prevalent_cases,
    ) = get_relevant_dfs()

    read_and_validate_text_for_clozapine_outcome(
        sorted_df,
        unsure_no_text_outcome_clozapine,
        unsure_validated_text_outcome_clozapine,
        prevalent_cases,
        non_prevalent_cases,
    )
