"""Test that the descriptive stats table is generated correctly."""


import time

import numpy as np
import pandas as pd
from psycop_model_evaluation.utils import bin_continuous_data
from tableone import TableOne


def test_generate_descriptive_stats_table(synth_eval_df: pd.DataFrame):
    """Test descriptive stats table."""
    df = synth_eval_df

    # Multiply the dataset by 10 to benchmark the performance
    df = pd.concat([df] * 10, ignore_index=True)

    # Assign 85% to train and 15% to test
    df["split"] = np.random.choice(
        ["1. train", "2. test", "3. val"],
        size=len(df),
        p=[0.85, 0.10, 0.05],
    )
    df["age_grouped"] = bin_continuous_data(
        series=df["age"],
        bins=[18, 35, 40, 45],
        bin_decimals=None,
    )[0]
    df = df.drop(columns=["age"])

    labels = {"is_female": "Female", "age_grouped": "Age group"}
    categorical = ["age_grouped", "is_female"]

    start_time = time.time()
    visits_table_df = TableOne(
        data=df,
        columns=list(labels),
        labels=labels,
        categorical=categorical,
        groupby="split",
        pval=True,
    ).tableone

    # Grouped
    df["dw_ek_borger"] = synth_eval_df["dw_ek_borger"]
    df["age"] = synth_eval_df["age"]
    grouped_df = df.groupby("dw_ek_borger").agg({"is_female": "max"})
    grouped_df["split"] = grouped_df.merge(df, how="left", on="dw_ek_borger")[["split"]]

    grouped_labels = {"is_female": "Female"}
    grouped_categorical_columns = ["is_female"]

    patients_table_df = TableOne(
        data=grouped_df,
        columns=list(grouped_labels),
        labels=grouped_labels,
        categorical=grouped_categorical_columns,
        groupby="split",
        pval=True,
    ).tableone
    end_time = time.time()

    round(end_time - start_time)

    pd.concat([visits_table_df, patients_table_df], axis=0)
