"""Assess model calibration level"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

from psycop.projects.forced_admission_inpatient.model_eval.config import BEST_POS_RATE
from psycop.projects.forced_admission_inpatient.model_eval.selected_runs import (
    get_best_eval_pipeline,
)
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    ForcedAdmissionInpatientPipelineRun,
)


def calibration_plot(run: ForcedAdmissionInpatientPipelineRun):
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(
        eval_ds.y, eval_ds.y_hat_probs, n_bins=30, strategy="uniform"
    )

    # only keep bins with a prob_pred value below 0.25
    prob_true = prob_true[prob_pred < 0.2]
    prob_pred = prob_pred[prob_pred < 0.2]

    # Create the subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    # Plot the calibration curve
    ax1.plot(
        prob_pred, prob_true, marker="o", color="#0072B2", label="Calibration Curve"
    )  # Calibration line color
    ax1.plot(
        [0, 0.2], [0, 0.2], linestyle="--", color="#D55E00", label="Perfectly Calibrated"
    )  # Dotted line color
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title("Calibration Curve")
    ax1.legend()
    ax1.axvline(x=BEST_POS_RATE, color="red", linestyle="--", label="Best Positive Rate")
    ax1.text(
    0.3,
    0.95,
    "Best positive predicted rate",
    color="red",
    fontsize=8,
    transform=ax1.transAxes,
    verticalalignment="top",
)


    # Plot the histogram with solid bars
    counts, bin_edges = np.histogram(eval_ds.y_hat_probs, bins=50, range=(0, 1))

    # Filter out bins with fewer than 5 entries
    filtered_counts = np.where(counts >= 5, counts, 0)
    ax2.bar(
        bin_edges[:-1],
        filtered_counts,
        width=np.diff(bin_edges),
        color="#0072B2",
        edgecolor="black",
    )
    ax2.text(
        0.6,
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
    ax2.axvline(x=BEST_POS_RATE, color="red", linestyle="--", label="Best Positive Rate")
    ax2.text(
    0.15,
    0.95,
    "Best positive predicted rate",
    color="red",
    fontsize=8,
    transform=ax2.transAxes,
    verticalalignment="top",
)

    # Adjust layout and save the figure
    plt.tight_layout()
    calibration_curve_hist_path = (
        run.paper_outputs.paths.figures / "fa_inpatient_calibration_curve_hist.png"
    )
    plt.savefig(calibration_curve_hist_path)
    plt.show()


if __name__ == "__main__":
    calibration_plot(run=get_best_eval_pipeline())
