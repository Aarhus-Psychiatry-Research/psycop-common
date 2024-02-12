from pathlib import Path

import polars as pl
from rapidfuzz import fuzz

from psycop.common.feature_generation.loaders.raw.load_text import load_text_sfis
from psycop.common.feature_generation.utils import write_df_to_file
from psycop.projects.clozapine.feature_generation.cohort_definition.eligible_prediction_times.schizophrenia_diagnosis import (
    add_only_patients_with_schizophrenia,
)


def add_only_patients_with_schizophrenia_with_text() -> pl.DataFrame:
    schizophrenia_df = add_only_patients_with_schizophrenia()

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
    relevant_text_df = pl.DataFrame(relevant_text_df)

    schizophrenia_df = schizophrenia_df.with_columns("dw_ek_borger").drop("timestamp")
    schizophrenia_df = schizophrenia_df.unique("dw_ek_borger")

    schizophrenia_text_medicin_df = relevant_text_df.join(
        schizophrenia_df, on="dw_ek_borger", how="inner"
    )

    return schizophrenia_text_medicin_df


def extract_clozapine_from_free_text() -> pl.DataFrame:
    clozapine_df = schizophrenia_text_medicin_df.clear()

    target_words = ["clozapin", "leponex"]

    # Iterate over rows in the DataFrame
    for index, value_str in enumerate(schizophrenia_text_medicin_df["value"]):
        value_words = value_str.split()

        for value in value_words:
            matches = []

            for target_word in target_words:
                # Find matches for the target word in the "value" column
                similarity_score = fuzz.ratio(value, target_word)

                # Check if the similarity score is above the specified cutoff
                if similarity_score > 60:
                    matches.append(target_word)

                    # Stop the inner loop if a match is found
                    break

            # If there are matches, add the row to the filtered DataFrame
            if matches:
                row_df = schizophrenia_text_medicin_df[index, :]

                # Concatenate the current row to the filtered DataFrame
                clozapine_df = pl.concat([clozapine_df, row_df])

                break

    return clozapine_df


if __name__ == "__main__":
    schizophrenia_text_medicin_df = add_only_patients_with_schizophrenia_with_text()

    clozapine_df = extract_clozapine_from_free_text()
    clozapine_df = clozapine_df.to_pandas()

    file_path = Path(
        "E:/shared_resources/clozapine/text_outcome/raw_text_outcome_clozapine.parquet"
    )

    write_df_to_file(df=clozapine_df, file_path=file_path)
