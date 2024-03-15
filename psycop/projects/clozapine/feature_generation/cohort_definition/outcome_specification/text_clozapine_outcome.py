import time
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz

from psycop.common.feature_generation.loaders.raw.load_text import load_text_sfis
from psycop.common.feature_generation.utils import write_df_to_file
from psycop.projects.clozapine.feature_generation.cohort_definition.eligible_prediction_times.schizophrenia_diagnosis import (
    add_only_patients_with_schizo,
)


def add_only_patients_with_schizo_with_text() -> pl.DataFrame:
    schizo_df = add_only_patients_with_schizo()

    relevant_sfi_names = [
        "Medicin",
        "Aktuelt psykisk",
        "Samtale med behandlingssigte",
        "Konklusion",
        "Vurdering/konklusion",
        "Plan",
        "Journalnotat",
        "Telefonnotat",
        "Telefonkonsultation",
    ]

    relevant_text_df = load_text_sfis(text_sfi_names=relevant_sfi_names, include_sfi_name=True)
    relevant_text_df["value"] = relevant_text_df["value"].str.lower()
    relevant_text_df = pl.DataFrame(relevant_text_df)

    schizo_df = schizo_df.with_columns("dw_ek_borger").drop("timestamp")
    schizo_df = schizo_df.unique("dw_ek_borger")

    schizo_text_medicin_df = relevant_text_df.join(schizo_df, on="dw_ek_borger", how="inner")

    return schizo_text_medicin_df


def extract_clozapine_from_free_text() -> pl.DataFrame:
    clozapine_df = schizophrenia_text_medicin_df.clear()

    # Add empty columns to clozapine_df
    clozapine_df = clozapine_df.with_columns(matched_word=pl.lit(""))
    clozapine_df = clozapine_df.with_columns(fuzz_ratio=pl.lit(0))

    target_words = ["clozapin", "leponex"]
    exclusion_list = [
        "Olanzapin",
        "olanzapin",
        "telefonen",
        "seponeres",
        "seponere",
        "seponers",
        "delepsine",
        "Delepsine",
        "alene",
        "seponeret",
        "pinex",
        "camping",
        "lempe",
        "loading",
        "Ancozan",
        "ancozan",
        "lene",
        "Lene",
        "cocain",
        "Lone",
        "lene",
        "Olazapin",
        "olazapin",
        "planzapin",
    ]

    # Set the number of rows to process before printing the time
    rows_per_print = 100000

    # Start the timer
    start_time = time.time()

    # Iterate over rows in the DataFrame
    for index, value_str in enumerate(schizophrenia_text_medicin_df["value"]):
        value_words = value_str.split()

        for value in value_words:
            matches = []

            # exclude words from exclusion list
            if value.lower() in exclusion_list:
                continue

            if value.lower().startswith("olanza"):
                continue  # Skip the row if it contains "olanza"

            if value.lower().startswith("telefo"):
                continue

            if value.lower().startswith("sepo"):
                continue

            if value.lower().startswith("ancoz"):
                continue

            for target_word in target_words:
                # Find matches for the target word in the "value" column
                similarity_score = fuzz.ratio(value, target_word)

                # Check if the similarity score is above the specified cutoff
                if similarity_score > 70:
                    matches.append(target_word)

                    # Stop the inner loop if a match is found
                    break

            # If there are matches, add the row to the filtered DataFrame
            if matches:
                row_df = schizophrenia_text_medicin_df[index, :]

                row_df = row_df.with_columns(matched_word=pl.lit(value))

                row_df = row_df.with_columns(fuzz_ratio=pl.lit(int(similarity_score)))  # type: ignore

                # Concatenate the current row to the filtered DataFrame
                clozapine_df = pl.concat([clozapine_df, row_df])

                break

        if (index + 1) % rows_per_print == 0:
            elapsed_time = time.time() - start_time  #
            print(f"Processed {index + 1} rows in {elapsed_time:.2f} seconds.")

    # Print the final elapsed time
    total_elapsed_time = time.time() - start_time
    print(f"Total processing time: {total_elapsed_time:.2f} seconds")

    return clozapine_df


if __name__ == "__main__":
    schizophrenia_text_medicin_df = add_only_patients_with_schizo_with_text()

    clozapine_df = extract_clozapine_from_free_text()
    clozapine_df = clozapine_df.to_pandas()

    file_path = Path(
        "E:/shared_resources/clozapine/text_outcome/raw_text_outcome_clozapine_v2.parquet"
    )

    write_df_to_file(df=clozapine_df, file_path=file_path)
