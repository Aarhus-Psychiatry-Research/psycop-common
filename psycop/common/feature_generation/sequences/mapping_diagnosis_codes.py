"""
To replicate the BEHRT paper, we need to map over diagnoses from ICD10 codes to caliber categories.
This script contains the functions for extracting that mapping.
"""

import os

import pandas as pd
import polars as pl


def extract_icd10_to_caliber_categories_df(in_path: str, out_path: str):
    """
    Extracts the mapping between ICD10 codes and caliber categories from the chronological-map-phenotypes repo.
    Saves the mapping as a csv file.
    Args:
        in_path (str): local path to folder 'chronological-map-phenotypes/secondary_care/'
        out_path (str): path for saving the output csv file
    """

    # Read all files in folder and concatenate them into one dataframe with polars
    df = pl.concat(
        [
            pl.read_csv(in_path + file)
            for file in os.listdir(in_path)
            if file.startswith("ICD")
        ],
    )

    # Keep only the disease and the icd10 code columns
    df = df.select(["Disease", "ICD10code"])

    # Remove non-alphanumeric characters from the icd10 column
    df = df.with_columns(
        pl.col("ICD10code").str.replace("[^a-zA-Z0-9]", ""),
    )

    # There are duplicates in the sense that the same icd10 code can be mapped to multiple diseases
    # Keep only the first row for each icd10 code. Consequence is that some diseases are lost, and
    # we only have 297 diseases instead of 301.
    df = df.sort("ICD10code")
    df = df.filter(pl.col("ICD10code").is_first())

    # Save the dataframe as a csv file
    df.write_csv(out_path + "caliber-icd10-mapping.csv")


def load_and_format_icd10_to_caliber_mapping(in_path: str) -> dict[str, str]:
    """
    Takes path to the .csv with the mapping between ICD10 codes and caliber categories and returns a dictionary with the mapping.
    This function is specific for mapping between ICD10 codes and caliber categories and can be swapped out for a different mapping.
    Args:
        in_path (str): Path to .csv file with the mapping between ICD10 codes and caliber categories.
    """
    mapping_df = pd.read_csv(in_path)
    mapping = mapping_df.set_index("ICD10code")["Disease"].to_dict()
    return mapping
