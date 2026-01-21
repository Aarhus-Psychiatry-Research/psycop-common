from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from psycop.projects.clozapine.model_eval.utils import read_eval_df_from_disk


def decision_curve_analysis(eval_df: pl.DataFrame) -> dict:  # type: ignore
    # Calculate calibration curve
    y_true = np.array(eval_df["y"])
    y_prob = np.array(eval_df["y_hat_prob"])

    # Define thresholds excluding 0 and 1 to avoid division by zero
    thresholds = np.linspace(0.01, 0.99, 100)

    # Number of samples
    n = len(y_true)

    # Prevalence of positive class
    treat_all = y_true.mean()

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


def plot_decision_curve(dca: dict, model_label: str | list[str]):  # type: ignore
    # Perform Decision Curve Analysis

    plt.figure(figsize=(8, 6))

    plt.plot(dca["thresholds"], dca["net_benefit_model"], label=model_label, color="#0072B2")
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

    plt.xlim(0, 0.2)
    plt.ylim(-0.2, 0.15)

    save_dir = Path("E:/shared_resources/clozapine/eval/figures")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"dca_clozapine_{model_label}.png"
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    # List of models to evaluate
    models = ["xgboost", "log_reg"]

    for model_label in models:
        # Build the experiment name including the current model
        experiment_name = (
            "clozapine hparam, structured_text_365d_lookahead, "
            f"{model_label}, 1 year lookbehind filter, 2025_random_split"
        )

        eval_dir = (
            f"E:/shared_resources/clozapine/eval_runs/"
            f"{experiment_name}_best_run_evaluated_on_test"
        )

        # Load the evaluation dataframe
        eval_df = read_eval_df_from_disk(eval_dir)

        # Compute DCA
        dca = decision_curve_analysis(eval_df=eval_df)

        # Plot and save single-model DCA
        plot_decision_curve(dca=dca, model_label=model_label)
