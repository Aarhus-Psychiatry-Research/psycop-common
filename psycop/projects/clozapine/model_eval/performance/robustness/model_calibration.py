"""Assess model calibration level and compute Brier score"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from psycop.projects.clozapine.model_eval.utils import read_eval_df_from_disk


def calibration_plot(eval_df: pd.DataFrame, save_dir: Path):
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(
        eval_df.y, eval_df["y_hat_prob"], n_bins=30, strategy="uniform"
    )

    positive_threshold = eval_df["y_hat_prob"].quantile(1 - best_pos_rate)

    # Only keep bins with a prob_pred value below 0.2
    prob_true = prob_true[prob_pred < 0.2]
    prob_pred = prob_pred[prob_pred < 0.2]

    # Compute Brier score
    brier = brier_score_loss(eval_df.y, eval_df["y_hat_prob"])

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))  # type: ignore

    # Plot calibration curve
    ax1.plot(prob_pred, prob_true, marker="o", color="#0072B2", label="Calibration Curve")
    ax1.plot([0, 0.2], [0, 0.2], linestyle="--", color="#D55E00", label="Perfectly Calibrated")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title(f"Calibration Curve (Brier Score = {brier:.4f})")
    ax1.legend()
    ax1.axvline(
        x=positive_threshold,
        color="red",
        linestyle="--",
        label=f"Positive prediction threshold (PPR of {best_pos_rate*100}%)",
    )
    ax1.text(
        0.31,
        0.95,
        f"Positive prediction threshold (PPR of {best_pos_rate*100}%)",
        color="red",
        fontsize=6,
        transform=ax1.transAxes,
        verticalalignment="top",
    )

    # Plot histogram of predicted probabilities
    counts, bin_edges = np.histogram(eval_df["y_hat_prob"], bins=50, range=(0, 1))
    filtered_counts = np.where(counts >= 5, counts, 0)
    ax2.bar(
        bin_edges[:-1],
        filtered_counts,
        width=np.diff(bin_edges),
        color="#0072B2",
        edgecolor="black",
    )
    ax2.text(
        0.65,
        0.95,
        "Bins with <5 entries have been removed",
        color="red",
        fontsize=8,
        transform=ax2.transAxes,
        verticalalignment="top",
    )
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Predicted Probabilities")
    ax2.axvline(
        x=positive_threshold,
        color="red",
        linestyle="--",
        label=f"Positive prediction threshold (PPR of {best_pos_rate*100}%)",
    )
    ax2.text(
        0.2,
        0.95,
        f"Positive prediction threshold (PPR of {best_pos_rate*100}%)",
        color="red",
        fontsize=8,
        transform=ax2.transAxes,
        verticalalignment="top",
    )

    # Adjust layout and save figure
    plt.tight_layout()
    calibration_curve_hist_path = save_dir / "clozapine_calibration_curve_hist_xgboost_7.5%.png"
    plt.savefig(calibration_curve_hist_path)
    plt.show()


if __name__ == "__main__":
    experiment_name = "clozapine hparam, structured_text_365d_lookahead, xgboost, 1 year lookbehind filter, 2025_random_split"
    best_pos_rate = 0.075
    eval_dir = (
        f"E:/shared_resources/clozapine/eval_runs/{experiment_name}_best_run_evaluated_on_test"
    )
    eval_df = read_eval_df_from_disk(eval_dir).to_pandas()

    save_dir = Path("E:/shared_resources/clozapine/eval/figures")
    calibration_plot(eval_df, save_dir)
