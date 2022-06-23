from pathlib import Path

import pandas as pd
import pytest

from psycopt2d.tables.performance_by_threshold import (
    days_from_positive_to_diagnosis,
    generate_performance_by_threshold_table,
    performance_by_threshold,
)


@pytest.fixture(scope="function")
def synth_data():
    csv_path = Path("tests") / "test_data" / "synth_data.csv"
    return pd.read_csv(csv_path)


def test_generate_performance_by_threshold_table(synth_data):
    df = synth_data

    table = generate_performance_by_threshold_table(
        labels=df["label"],
        pred_probs=df["pred_prob"],
        ids=df["dw_ek_borger"],
        threshold_percentiles=[
            0.99,
            0.95,
        ],
        pred_timestamps=df["timestamp"],
        outcome_timestamps=df["timestamp_t2d_diag"],
        output_format="df",
    )

    expected_df = pd.DataFrame(
        {
            "threshold_percentile": {0: 0.99, 1: 0.95},
            "prevalence": {0: 0.01, 1: 0.05},
            "PPV": {0: 0.06, 1: 0.05},
            "NPV": {0: 0.95, 1: 0.95},
            "sensitivity": {0: 0.01, 1: 0.05},
            "specificity": {0: 0.99, 1: 0.95},
            "FPR": {0: 0.01, 1: 0.05},
            "FNR": {0: 0.99, 1: 0.95},
            "accuracy": {0: 0.94, 1: 0.9},
            "false_positives": {0: 940, 1: 4757},
            "warning_days": {0: 70822, 1: 297274},
            "warning_days_per_false_positive": {0: 62.49, 1: 62.49},
        },
    )

    for col in table.columns:
        table[col].equals(expected_df[col])


def test_diag_characteristics_by_threshold(synth_data):
    df = synth_data

    matrices = {
        f"threshold_{threshold/10}": performance_by_threshold(
            labels=df["label"],
            pred_probs=df["pred_prob"],
            positive_threshold=threshold / 10,
        )
        for threshold in range(0, 5, 1)
    }

    assert matrices["threshold_0.1"]["prevalence"][0] == 0.45
    assert matrices["threshold_0.2"]["prevalence"][0] == 0.35
    assert matrices["threshold_0.3"]["prevalence"][0] == 0.25
    assert matrices["threshold_0.4"]["prevalence"][0] == 0.15


def test_time_from_flag_to_diag(synth_data):
    df = synth_data

    # Threshold = 0.5
    assert (
        days_from_positive_to_diagnosis(
            ids=df["dw_ek_borger"],
            pred_probs=df["pred_prob"],
            pred_timestamps=df["timestamp"],
            outcome_timestamps=df["timestamp_t2d_diag"],
            positive_threshold=0.5,
        )
        == 290996
    )

    # Threshold = 0.2
    assert (
        days_from_positive_to_diagnosis(
            ids=df["dw_ek_borger"],
            pred_probs=df["pred_prob"],
            pred_timestamps=df["timestamp"],
            outcome_timestamps=df["timestamp_t2d_diag"],
            positive_threshold=0.2,
        )
        == 1_878_900
    )
