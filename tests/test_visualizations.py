from pathlib import Path

import altair as alt
import pandas as pd

from psycopt2d.visualization import plot_prob_over_time


def test_prob_over_time():
    """Test visualization runs on test set."""
    repo_path = Path(__file__).parent
    path = repo_path / "test_data" / "synth_data.csv"
    df = pd.read_csv(path)
    alt.data_transformers.disable_max_rows()

    plot_prob_over_time(
        patient_id=df["dw_ek_borger"],
        timestamp=df["timestamp"],
        pred_prob=df["pred_prob"],
        outcome_timestamp=df["timestamp_t2d_diag"],
        label=df["label"],
        look_behind_distance=500,
    )
