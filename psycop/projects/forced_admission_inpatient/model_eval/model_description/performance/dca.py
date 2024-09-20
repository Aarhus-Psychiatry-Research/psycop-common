from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from psycop.projects.forced_admission_inpatient.model_eval.config import (
    BEST_POS_RATE,
    DEVELOPMENT_GROUP,
    MODEL_ALGORITHM,
)
from psycop.projects.forced_admission_inpatient.model_eval.run_pipeline_on_val import (
    test_selected_model_pipeline,
)
from psycop.projects.forced_admission_inpatient.model_eval.selected_runs import (
    get_best_eval_pipeline,
)
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    ForcedAdmissionInpatientPipelineRun,
)


def decision_curve_analysis(run: ForcedAdmissionInpatientPipelineRun):
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    # Calculate calibration curve
    y_true, y_prob = eval_ds.y, eval_ds.y_hat_probs

    # Define thresholds excluding 0 and 1 to avoid division by zero
    thresholds = np.linspace(0.01, 0.99, 100)

    # Number of samples
    n = len(y_true)

    # Prevalence of positive class
    treat_all = np.mean(y_true)

    # Net benefits for different strategies
    net_benefit_model = []
    net_benefit_all = []
    net_benefit_none = [0] * len(thresholds)  # Net benefit for treating none is always 0

    # Iterate over thresholds to calculate net benefits
    for threshold in thresholds:
        # Predicted positive based on threshold
        y_pred = (y_prob >= threshold).astype(int)

        # True Positives and False Positives
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))

        # Net benefit calculation for the model
        nb_model = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        nb_all = treat_all - (1 - treat_all) * (threshold / (1 - threshold))

        # Append net benefits
        net_benefit_model.append(nb_model)
        net_benefit_all.append(nb_all)

    # Package dca in a dictionary
    dca = {
        "thresholds": thresholds,
        "net_benefit_model": net_benefit_model,
        "net_benefit_all": net_benefit_all,
        "net_benefit_none": net_benefit_none,
    }

    return dca


def plot_decision_curve(run: ForcedAdmissionInpatientPipelineRun):
    # Perform Decision Curve Analysis
    dca = decision_curve_analysis(run=run)

    plt.figure(figsize=(8, 6))

    plt.plot(
        dca["thresholds"],
        dca["net_benefit_model"],
        label="XGBoost" if run.model_type == "xgboost" else "ElasticNet",
        color="#0072B2",
    )
    plt.plot(
        dca["thresholds"],
        dca["net_benefit_all"],
        label="Treat All",
        linestyle="--",
        color="#009E73",
    )
    plt.plot(
        dca["thresholds"], dca["net_benefit_none"], label="Treat None", linestyle="--", color="gray"
    )

    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis")
    plt.legend(loc="best")
    plt.grid(True)

    # Set axis limits
    plt.xlim(0, 0.2)  # X-axis limit from 0 to 0.2
    plt.ylim(-0.2, 0.15)  # Y-axis limit from -1 to 1 (or adjust upper bound if needed)

    # Save and show the plot
    dca_path = run.paper_outputs.paths.figures / "fa_inpatient_dca.png"
    plt.savefig(dca_path)


def plot_decision_curve_multiple_runs(runs: Sequence[ForcedAdmissionInpatientPipelineRun]):
    dca = decision_curve_analysis(run=runs[0])

    dca["net_benefit_model_1"] = dca.pop("net_benefit_model")

    for i, run in enumerate(runs[1:], 1):
        dca_new = decision_curve_analysis(run=run)

        # Append net_benefit_model to the dca dictionary as net_benefit_model_1, net_benefit_model_2, etc.
        dca[f"net_benefit_model_{i+1}"] = dca_new["net_benefit_model"]

    # create a colour palette for the model lines
    palette = plt.cm.get_cmap("tab10", len(runs) + 1)

    plt.figure(figsize=(8, 6))

    plt.plot(
        dca["thresholds"],
        dca["net_benefit_all"],
        label="Treat All",
        linestyle="--",
        color="#009E73",
    )
    plt.plot(
        dca["thresholds"], dca["net_benefit_none"], label="Treat None", linestyle="--", color="gray"
    )
    # plot the model lines
    for i, run in enumerate(runs, 1):
        # Add label to model lines
        model_name = "XGBoost" if run.model_type == "xgboost" else "ElasticNet"

        plt.plot(
            dca["thresholds"],
            dca[f"net_benefit_model_{i}"],
            label=f"{model_name}",
            color=palette(i),
            linewidth=1,
            alpha=0.8,
        )

    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis")
    plt.legend(loc="best")
    plt.grid(True)

    # Set axis limits
    plt.xlim(0, 0.2)  # X-axis limit from 0 to 0.2
    plt.ylim(-0.2, 0.15)  # Y-axis limit from -1 to 1 (or adjust upper bound if needed)

    # Save and show the plot
    EVAL_ROOT = Path("E:/shared_resources/forced_admissions_inpatient/eval")
    dca_path = EVAL_ROOT / "fa_inpatient_dca_all_models.png"
    plt.savefig(dca_path)


if __name__ == "__main__":
    # Plot the Decision Curve
    plot_decision_curve(run=get_best_eval_pipeline())

    xgboost = ForcedAdmissionInpatientPipelineRun(
        group=DEVELOPMENT_GROUP,
        name=DEVELOPMENT_GROUP.get_best_runs_by_lookahead()[1, 2],
        pos_rate=BEST_POS_RATE,
        create_output_paths_on_init=False,
    )
    xgboost_eval = test_selected_model_pipeline(
        pipeline_to_test=xgboost,
        splits_for_training=["train", "val"],
        splits_for_evaluation=["test"],  # add with_washout if eval on cohort with washout
    )
    lr = ForcedAdmissionInpatientPipelineRun(
        group=DEVELOPMENT_GROUP,
        name=DEVELOPMENT_GROUP.get_best_runs_by_lookahead()[0, 2],
        pos_rate=BEST_POS_RATE,
        create_output_paths_on_init=False,
    )

    lr_eval = test_selected_model_pipeline(
        pipeline_to_test=lr,
        splits_for_training=["train", "val"],
        splits_for_evaluation=["test"],  # add with_washout if eval on cohort with washout
    )

    # Plot the Decision Curve for the best XGBoost model AND the best Logistic Regression model
    plot_decision_curve_multiple_runs(runs=[xgboost_eval, lr_eval])
