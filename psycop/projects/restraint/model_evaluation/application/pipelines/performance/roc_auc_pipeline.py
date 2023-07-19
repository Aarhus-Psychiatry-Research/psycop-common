from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as pn

from psycop.common.model_evaluation.binary.global_performance.roc_auc import (
    bootstrap_roc,
)
from psycop.projects.restraint.model_evaluation.config import (
    EVAL_RUN,
    FIGURES_PATH,
    MODEL_NAME,
    PN_THEME,
    TEXT_EVAL_RUN,
    TEXT_FIGURES_PATH,
)
from psycop.projects.restraint.utils.best_runs import Run


def bootstrap_results(
    y: pd.Series,
    y_hat_probs: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tprs_bootstrapped, aucs_bootstrapped, base_fpr = bootstrap_roc(
        n_bootstraps=1000,
        y=y,
        y_hat_probs=y_hat_probs,
    )

    return tprs_bootstrapped, aucs_bootstrapped, base_fpr


def plot_auc_roc(
    tprs_bootstrapped: np.ndarray,
    aucs_bootstrapped: np.ndarray,
    base_fpr: np.ndarray,
    title: str = "Receiver Operating Characteristic (ROC) Curve",
) -> pn.ggplot:
    # We need a custom bootstrap implementation, because using scipy.bootstrap
    # on the roc_curve method will yield different fpr values for each resample,
    # and thus the tpr values will be interpolated on different fpr values. This
    # will result in arrays of different dimensions, which will cause an error.

    mean_tprs = tprs_bootstrapped.mean(axis=0)
    se_tprs = tprs_bootstrapped.std(axis=0) / np.sqrt(1000)

    # Calculate confidence interval for TPR over all FPRs
    tprs_upper = mean_tprs + se_tprs
    tprs_lower = mean_tprs - se_tprs

    # Calculate confidence interval for AUC
    auc_mean = np.mean(aucs_bootstrapped)
    auc_se = np.std(aucs_bootstrapped) / np.sqrt(1000)
    auc_ci = [auc_mean - 1.96 * auc_se, auc_mean + 1.96 * auc_se]

    df = pd.DataFrame(
        {
            "fpr": base_fpr,
            "tpr": mean_tprs,
            "tpr_lower": tprs_lower,
            "tpr_upper": tprs_upper,
        },
    )

    auroc_label = pn.annotate(
        "text",
        label=f"AUROC (95% CI): {auc_mean:.3f} ({auc_ci[0]:.3f}-{auc_ci[1]:.3f})",
        x=1,
        y=0,
        ha="right",
        va="bottom",
        size=18,
    )

    # Plot AUC ROC curve
    return (
        pn.ggplot(df, pn.aes(x="fpr", y="tpr"))
        + pn.geom_line(size=1)
        + pn.labs(title=title, x="1 - Specificity", y="Sensitivity")
        + pn.xlim(0, 1)
        + pn.ylim(0, 1)
        + PN_THEME
        + pn.theme(
            axis_text=pn.element_text(size=15),
            axis_title=pn.element_text(size=18),
            plot_title=pn.element_text(size=22),
            legend_position="none",
        )
        + auroc_label
    )


def roc_auc_pipeline(run: Run, path: Path):
    eval_ds = run.get_eval_dataset()

    if isinstance(eval_ds.y, pd.DataFrame) or isinstance(
        eval_ds.y_hat_probs,
        pd.DataFrame,
    ):
        raise TypeError

    tprs_bootstrapped, aucs_bootstrapped, base_fpr = bootstrap_results(
        y=eval_ds.y,
        y_hat_probs=eval_ds.y_hat_probs,
    )

    # Calculate confidence interval for AUC
    auc_mean = np.mean(aucs_bootstrapped)
    auc_se = np.std(aucs_bootstrapped) / np.sqrt(1000)
    auc_ci = [auc_mean - 1.96 * auc_se, auc_mean + 1.96 * auc_se]

    print(f"AUROC (95% CI): {auc_mean} ({auc_ci[0]}-{auc_ci[1]})")

    auc = plot_auc_roc(
        tprs_bootstrapped=tprs_bootstrapped,
        aucs_bootstrapped=aucs_bootstrapped,
        base_fpr=base_fpr,
        title=f"AUROC for {MODEL_NAME[run.name]}",
    )

    auc.save(path / "roc_auc.png", dpi=300)


if __name__ == "__main__":
    roc_auc_pipeline(EVAL_RUN, FIGURES_PATH)
    roc_auc_pipeline(TEXT_EVAL_RUN, TEXT_FIGURES_PATH)
