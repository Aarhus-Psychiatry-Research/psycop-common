from pathlib import Path

import pandas as pd
import pytest

from psycopt2d.visualization.performance_by_threshold import (
    get_time_from_pos_to_diag_df,
    performance_by_threshold,
)


@pytest.fixture(scope="function")
def synth_data():
    csv_path = Path("tests") / "test_data" / "synth_data.csv"
    return pd.read_csv(csv_path)


def test_diag_characteristics_by_threshold(synth_data):
    df = synth_data

    matrices = {
        f"threshold_{threshold/10}": performance_by_threshold(
            real_values=df["label"],
            pred_probs=df["pred_prob"],
            threshold=threshold / 10,
        )
        for threshold in range(0, 5, 1)
    }

    assert matrices["threshold_0.1"]["prevalence"] == 0.45
    assert matrices["threshold_0.2"]["prevalence"] == 0.35
    assert matrices["threshold_0.3"]["prevalence"] == 0.25
    assert matrices["threshold_0.4"]["prevalence"] == 0.15


def test_time_from_flag_to_diag(synth_data):
    # Threshold = 0.5
    time_from_pos_to_diag_df = get_time_from_pos_to_diag_df(
        eval_df=synth_data,
        positive_threshold=0.5,
    )

    assert time_from_pos_to_diag_df["undiagnosed_days_saved"] == 290996
    assert time_from_pos_to_diag_df["days_saved_per_true_positive"] == 1238.3

    # Threshold = 0.2
    time_from_pos_to_diag_df = get_time_from_pos_to_diag_df(
        eval_df=synth_data,
        positive_threshold=0.2,
    )

    assert time_from_pos_to_diag_df["undiagnosed_days_saved"] == 1_878_900
    assert time_from_pos_to_diag_df["days_saved_per_true_positive"] == 1_289.6
