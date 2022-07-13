from typing import Iterable

import altair as alt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from psycopt2d.utils import bin_continuous_data, round_floats_to_edge


def create_auc_from_first_visit_df(
    labels: Iterable[int],
    y_hat_probs: Iterable[int],
    first_visit_timestamps: Iterable[pd.Timestamp],
    prediction_timestamps: Iterable[pd.Timestamp],
    bins=[0, 1, 7, 14, 28, 182, 365, 730, 1825],
    pretty_bins: bool = True,
) -> pd.DataFrame:
    """Calculate auc by time from first visit.

    Args:
        labels (Iterable[int]): True labels
        y_hat_probs (Iterable[int]): Predicted probabilities
        first_visit_timestamps (Iterable[pd.Timestamp]): Timestamp of first visit
        prediction_timestamps (Iterable[pd.Timestamp]): Timestamp of prediction
        bins (list, optional): Bins to group by. Defaults to [0, 1, 7, 14, 28, 182, 365, 730, 1825].
        pretty_bins (bool): Whether to format bins nicely

    Returns:
        pd.DataFrame: Dataframe ready for plotting
    """

    df = pd.DataFrame(
        {
            "y": labels,
            "y_hat": y_hat_probs,
            "first_visit_timestamp": first_visit_timestamps,
            "prediction_timestamp": prediction_timestamps,
        },
    )

    # Calculate difference in days between prediction and first visit
    df["days_from_first_visit"] = (
        df["prediction_timestamp"] - df["first_visit_timestamp"]
    ) / np.timedelta64(1, "D")

    # bin data
    bin_fn = bin_continuous_data if pretty_bins else round_floats_to_edge
    df["days_from_first_visit_binned"] = bin_fn(df["days_from_first_visit"], bins=bins)

    def _calc_auc(df: pd.DataFrame):
        return roc_auc_score(df["y"], df["y_hat"])

    # Calc AUC and prettify output
    output_df = df.groupby("days_from_first_visit_binned").apply(_calc_auc)
    output_df = output_df.reset_index().rename({0: "AUC"}, axis=1)
    return output_df


def plot_auc_time_from_first_visit(
    labels: Iterable[int],
    y_hat_probs: Iterable[int],
    first_visit_timestamps: Iterable[pd.Timestamp],
    prediction_timestamps: Iterable[pd.Timestamp],
    bins=[0, 1, 7, 14, 28, 182, 365, 730, 1825],
    pretty_bins: bool = True,
) -> alt.Chart:

    df = create_auc_from_first_visit_df(
        labels=labels,
        y_hat_probs=y_hat_probs,
        first_visit_timestamps=first_visit_timestamps,
        prediction_timestamps=prediction_timestamps,
        bins=bins,
        pretty_bins=pretty_bins,
    )
    sort_order = np.arange(len(df))
    return plot_bar_chart(
        x_values=df["days_from_first_visit_binned"],
        y_values=df["AUC"],
        x_title="Days from first visit",
        y_title="AUC",
        sort=sort_order,
    )


if __name__ == "__main__":
    from pathlib import Path

    from psycopt2d.visualization.base_charts import plot_bar_chart

    repo_path = Path(__file__).parent.parent.parent.parent
    path = repo_path / "tests" / "test_data" / "synth_eval_data.csv"
    df = pd.read_csv(path)
    df.head()
    alt.data_transformers.disable_max_rows()

    perf = create_auc_from_first_visit_df(
        labels=df["label"],
        y_hat_probs=df["pred_prob"],
        first_visit_timestamps=pd.to_datetime(df["timestamp_first_pred_time"]),
        prediction_timestamps=pd.to_datetime(df["timestamp"]),
        pretty_bins=False,
    )
    plot_bar_chart(
        x_values=perf["days_from_first_visit_binned"],
        y_values=perf["AUC"],
        x_title="Days from first visit",
        y_title="AUC",
        sort=np.arange(len(perf)),
    )
