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
    csv_path = Path("tests") / "test_data" / "synth_eval_data.csv"
    return pd.read_csv(csv_path)


def test_generate_performance_by_threshold_table(synth_data):
    df = synth_data

    table = generate_performance_by_threshold_table(
        labels=df["label"],
        pred_probs=df["pred_prob"],
        ids=df["dw_ek_borger"],
        threshold_percentiles=[0.99, 0.5, 0.01],
        pred_timestamps=df["timestamp"],
        outcome_timestamps=df["timestamp_t2d_diag"],
        output_format="df",
    )

    expected_df = pd.DataFrame(
        {
            "threshold_percentile": {0: 0.99, 1: 0.5, 2: 0.01},
            "true_prevalence": {0: 0.0502, 1: 0.0502, 2: 0.0502},
            "positive_rate": {0: 0.01, 1: 0.5, 2: 0.5511},
            "negative_rate": {0: 0.99, 1: 0.5, 2: 0.4489},
            "PPV": {0: 0.06, 1: 0.0502, 2: 0.0502},
            "NPV": {0: 0.9499, 1: 0.9497, 2: 0.9497},
            "sensitivity": {0: 0.0119, 1: 0.4997, 2: 0.5503},
            "specificity": {0: 0.9901, 1: 0.5, 2: 0.4488},
            "FPR": {0: 0.0099, 1: 0.5, 2: 0.5512},
            "FNR": {0: 0.9881, 1: 0.5003, 2: 0.4497},
            "accuracy": {0: 0.941, 1: 0.5, 2: 0.4539},
            "warning_days": {0: 70822, 1: 2618039, 2: 4609550},
            "warning_days_per_false_positive": {0: 88.05, 1: 88.05, 2: 88.05},
            "true_positives": {0: 60, 1: 2510, 2: 2764},
            "true_negatives": {0: 94037, 1: 47487, 2: 42627},
            "false_positives": {0: 940, 1: 47490, 2: 52350},
            "false_negatives": {0: 4963, 1: 2513, 2: 2259},
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
