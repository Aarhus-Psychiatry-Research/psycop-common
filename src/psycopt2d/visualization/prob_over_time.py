"""Code for creating probabilities over time plots."""

from datetime import datetime
from typing import Iterable, Optional, Union

import altair as alt
import pandas as pd


def plot_prob_over_time(
    timestamp: Iterable[datetime],
    pred_prob: Iterable[float],
    label: Iterable[Union[int, str]],
    outcome_timestamp: Iterable[Union[datetime, None]],
    patient_id: Iterable[Union[int, str]],
    x_axis: str = "Time from outcome",
    y_axis: str = "Model Predictive Probability",
    legend: str = "Highest Predictive Probability",
    look_behind_distance: Optional[int] = None,
    line_opacity: float = 0.3,
) -> alt.Chart:
    """Plot probabilities over time for a given outcome. Each element passed
    (e.g. timestamp, pred_prob etc.) must have the same length, and for each
    iterable, the i'th item must correspond to the same patient.

    Args:
        timestamp (Iterable[datetime]): Timestamps for each prediction time.
        pred_prob (Iterable[float]): The predictive probabilities of the model for each prediction time.
        label (Iterable[Union[int, str]]): True labels for each prediction time.
        outcome_timestamp (Iterable[Union[datetime, None]]): Timestamp of the
            positive outcome.
        patient_id (Iterable[Union[int, str]]): Patient ID for each prediction time. Used for
            connecting timestamp/pred-prob points into one line pr. patient.
        x_axis (str, optional): Label on x-axis. Defaults to "Time from outcome".
        y_axis (str, optional): Label of y-axis. Defaults to "Model Predictive
            Probability".
        legend (str, optional): Label on legend. Defaults to "Highest Predictive
            Probability".
        look_behind_distance (Optional[int], optional): Look-behind window. Used for
            shading the corresponding area. Defaults to None in which case no shaded
            areas is plotted.

    Returns:
        alt.Chart: An altair chart object.

    Examples:
        >>> from pathlib import Path
        >>> repo_path = Path(__file__).parent.parent.parent.parent
        >>> path = repo_path / "tests" / "test_data" / "synth_data.csv"
        >>> df = pd.read_csv(path)
        >>> alt.data_transformers.disable_max_rows()

        >>> plot_prob_over_time(
        >>>     patient_id=df["dw_ek_borger"],
        >>>     timestamp=df["timestamp"],
        >>>     pred_prob=df["pred_prob"],
        >>>     outcome_timestamp=df["timestamp_t2d_diag"],
        >>>     label=df["label"],
        >>>     look_behind=500,
        >>> )
    """

    # construct pandas df ensuring types
    plot_df = pd.DataFrame(
        {
            "timestamp": list(timestamp),
            "pred_prob": list(pred_prob),
            "outcome_timestamp": list(outcome_timestamp),
            "patient_id": list(patient_id),
            "label": list(label),
        },
    )
    # remove individuals with no outcome
    plot_df = plot_df.dropna()

    time_cols = ["timestamp", "outcome_timestamp"]
    plot_df.loc[:, time_cols] = plot_df[time_cols].apply(pd.to_datetime)
    plot_df["delta_time"] = plot_df["timestamp"] - plot_df["outcome_timestamp"]
    plot_df["delta_time"] = plot_df["delta_time"].dt.days

    plot_df["patient_id"] = plot_df["patient_id"].astype(str)
    plot_df["label"] = plot_df["label"].astype(str)

    max_pred_prob = plot_df.groupby(["patient_id"])["pred_prob"].max()
    plot_df["color"] = [max_pred_prob[id_] for id_ in plot_df["patient_id"]]

    chart = (
        alt.Chart(plot_df)
        .mark_line(opacity=line_opacity)
        .encode(
            x=alt.X("delta_time", axis=alt.Axis(title=x_axis)),
            y=alt.Y("pred_prob", axis=alt.Axis(format="%", title=y_axis)),
            color=alt.Color(
                "color",
                legend=alt.Legend(title=legend),
                scale=alt.Scale(scheme="plasma"),
            ),
            detail="patient_id",
        )
    )

    if look_behind_distance:
        areas_cutoffs = pd.DataFrame(
            {
                "start": [0 - look_behind_distance],
                "stop": [0],
                "Predictive Window": "Positive",
            },
        ).reset_index()

        areas = (
            alt.Chart(areas_cutoffs)
            .mark_rect(opacity=0.2)
            .encode(
                x="start",
                x2="stop",
                y=alt.value(0),  # pixels from top
                y2=alt.value(300),  # pixels from top
                color="Predictive Window",
            )
        )
        chart = areas + chart
    return chart


if __name__ == "__main__":
    from pathlib import Path

    repo_path = Path(__file__).parent.parent.parent.parent
    path = repo_path / "tests" / "test_data" / "synth_data.csv"
    df = pd.read_csv(path)
    df.head()
    alt.data_transformers.disable_max_rows()

    plot_prob_over_time(
        patient_id=df["dw_ek_borger"],
        timestamp=df["timestamp"],
        pred_prob=df["pred_prob"],
        outcome_timestamp=df["timestamp_t2d_diag"],
        label=df["label"],
        look_behind_distance=500,
    )
