# Find all instances of positive urine samples that are followed by/preceeded by administration UVI-related antibiotics

# Join to admission id and keep only first positive outcome during each admission
from psycop.common.feature_generation.loaders_2024.load_uti_relevant_anitbiotics import uti_relevant_antibiotics
from psycop.common.feature_generation.loaders_2024.load_urine_samples import uti_positive_urine_samples

import pandas as pd
import numpy as np

def uti_outcomes():
    
    # load data
    antibiotics_df = uti_relevant_antibiotics()
    
    antibiotics_df = antibiotics_df.rename(columns={"timestamp": "administration_time"})

    postive_urine_samples_df = uti_positive_urine_samples()

    postive_urine_samples_df = postive_urine_samples_df.rename(columns={"timestamp": "sample_time"})

    # merge the DataFrames on patient_id
    merged_df = postive_urine_samples_df.merge(antibiotics_df, on='dw_ek_borger', how='inner')


    # Calculate the time difference in days
    merged_df['time_diff'] = (merged_df['administration_time'] - merged_df['sample_time']).dt.days

    # Filter for antibiotic prescriptions between 1 day before and 5 days after the sample date
    filtered_df = merged_df[(merged_df['time_diff'] >= -1) & (merged_df['time_diff'] <= 5)]

    filtered_df = filtered_df.rename(columns={"sample_time": "timestamp", "value_x": "value"})    

    return filtered_df[["dw_ek_borger", "timestamp"]]


if __name__ == "__main__":
    df = uti_outcomes()
    print('Hi!')