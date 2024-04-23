"""Assess model calibration level"""

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from psycop.projects.forced_admission_outpatient.model_eval.selected_runs import (
    get_best_eval_pipeline,
)
from psycop.projects.forced_admission_outpatient.utils.pipeline_objects import (
    ForcedAdmissionOutpatientPipelineRun,
)


def calibration_plot(run: ForcedAdmissionOutpatientPipelineRun):
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(
        eval_ds.y, eval_ds.y_hat_probs, n_bins=10, strategy="uniform"
    )

    # Plot the calibration curve
    plt.plot(prob_pred, prob_true, marker="o", label="Calibration Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    calibration_curve_path = run.paper_outputs.paths.figures / "fa_outpatient_calibration_curve.png"
    plt.savefig(calibration_curve_path)
    plt.show()


if __name__ == "__main__":
    calibration_plot(run=get_best_eval_pipeline())
