# Find all instances of positive urine samples that are followed by/preceeded by administration UVI-related antibiotics

# Join to admission id and keep only first positive outcome during each admission
import pandas as pd

from psycop.common.cohort_definition import OutcomeTimestampFrame
from psycop.common.feature_generation.loaders_2024.load_urine_samples import (
    uti_positive_urine_samples,
)
from psycop.common.feature_generation.loaders_2024.load_uti_relevant_anitbiotics import (
    uti_relevant_antibiotics,
)
from psycop.common.global_utils.cache import shared_cache


@shared_cache().cache()
def uti_outcome_timestamps() -> pd.DataFrame:
    return uti_outcomes()

@shared_cache().cache()
def uti_postive_urine_sample_outcome_timestamps() -> pd.DataFrame:
    return uti_postive_urine_sample_outcomes()

@shared_cache().cache()
def uti_relevant_antibiotics_administrations_outcome_timestamps() -> pd.DataFrame:
    return uti_relevant_antibiotics_outcomes()

def uti_outcomes() -> pd.DataFrame:
    # load data
    antibiotics_df = uti_relevant_antibiotics()

    antibiotics_df = antibiotics_df.rename(columns={"timestamp": "administration_time"})

    postive_urine_samples_df = uti_positive_urine_samples()

    postive_urine_samples_df = postive_urine_samples_df.rename(columns={"timestamp": "sample_time"})

    # merge the DataFrames on patient_id
    merged_df = postive_urine_samples_df.merge(antibiotics_df, on="dw_ek_borger", how="inner")

    # Calculate the time difference in days
    merged_df["time_diff"] = (merged_df["administration_time"] - merged_df["sample_time"]).dt.days

    # Filter for antibiotic prescriptions between 1 day before and 5 days after the sample date
    filtered_df = merged_df[(merged_df["time_diff"] >= -1) & (merged_df["time_diff"] <= 5)]

    # Create sample uuid 
    filtered_df['sample_uuid'] = filtered_df.dw_ek_borger.astype(str) + filtered_df.sample_time.astype(str)

    # Keep only one row per sample
    filtered_df = filtered_df.drop_duplicates(subset='sample_uuid')
    
    filtered_df = filtered_df.rename(columns={"sample_time": "timestamp", "afsnit_administration": "value",})

    return filtered_df[["dw_ek_borger", "timestamp", "value",]]


def uti_postive_urine_sample_outcomes() -> pd.DataFrame:

    df = uti_positive_urine_samples()[["dw_ek_borger", "timestamp", "afdeling_rekivernt"]]

    df = df.rename(columns={"afdeling_rekivernt": "value"})
    
    return df

def uti_relevant_antibiotics_outcomes() -> pd.DataFrame:
    
    df = uti_relevant_antibiotics()[["dw_ek_borger", "timestamp", "afsnit_administration"]]
    
    df = df.rename(columns={"afsnit_administration": "value"})

    return df

if __name__ == "__main__":
    df = uti_outcomes()
    print("Hi!")
