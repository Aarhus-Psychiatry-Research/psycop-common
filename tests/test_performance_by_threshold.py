from pathlib import Path

import pandas as pd
import pytest

from psycopt2d.tables.performance_by_threshold import (
    days_from_positive_to_diagnosis,
    generate_performance_by_threshold_table,
)


@pytest.fixture(scope="function")
def synth_data():
    csv_path = Path("tests") / "test_data" / "synth_eval_data.csv"
    df = pd.read_csv(csv_path)

    # Convert all timestamp cols to datetime
    for col in [col for col in df.columns if "timestamp" in col]:
        df[col] = pd.to_datetime(df[col])

    return df


def test_generate_performance_by_threshold_table(synth_data):
    df = synth_data

    table = generate_performance_by_threshold_table(
        labels=df["label"],
        pred_probs=df["pred_prob"],
        ids=df["dw_ek_borger"],
        threshold_percentiles=[0.2, 0.1],
        pred_timestamps=df["timestamp"],
        outcome_timestamps=df["timestamp_t2d_diag"],
        output_format="df",
    )

    expected_df = pd.DataFrame(
        {
            "threshold_percentile": {0: 0.2, 1: 0.1},
            "true_prevalence": {0: 0.0502, 1: 0.0502},
            "positive_rate": {0: 0.3497, 1: 0.4517},
            "negative_rate": {0: 0.6503, 1: 0.5483},
            "PPV": {0: 0.0491, 1: 0.0501},
            "NPV": {0: 0.9492, 1: 0.9497},
            "sensitivity": {0: 0.3418, 1: 0.4507},
            "specificity": {0: 0.6499, 1: 0.5483},
            "FPR": {0: 0.3501, 1: 0.4517},
            "FNR": {0: 0.6582, 1: 0.5493},
            "accuracy": {0: 0.6344, 1: 0.5434},
            "true_positives": {0: 1717, 1: 2264},
            "true_negatives": {0: 61728, 1: 52074},
            "false_positives": {0: 33249, 1: 42903},
            "false_negatives": {0: 3306, 1: 2759},
            "warning_days": {0: 1880143.0, 1: 2412489.0},
            "warning_days_per_false_positive": {0: 56.23, 1: 56.23},
        },
    )

    for col in table.columns:
        table[col].equals(expected_df[col])


def test_time_from_flag_to_diag(synth_data):
    df = synth_data

    # Threshold = 0.5
    val = days_from_positive_to_diagnosis(
        ids=df["dw_ek_borger"],
        pred_probs=df["pred_prob"],
        pred_timestamps=df["timestamp"],
        outcome_timestamps=df["timestamp_t2d_diag"],
        positive_threshold=0.5,
    )

    assert val > 290_000 and val < 292_000

    # Threshold = 0.2
    val = days_from_positive_to_diagnosis(
        ids=df["dw_ek_borger"],
        pred_probs=df["pred_prob"],
        pred_timestamps=df["timestamp"],
        outcome_timestamps=df["timestamp_t2d_diag"],
        positive_threshold=0.2,
    )

    assert val > 1_875_000 and val < 1_885_000
