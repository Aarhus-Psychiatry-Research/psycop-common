"""AUC ROC curve."""

import numpy as np
import pandas as pd
import plotnine as pn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import resample

from psycop.common.model_training.training_output.dataclasses import EvalDataset


def bootstrap_roc(
    n_bootstraps: int,
    y: pd.Series,
    y_hat_probs: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tprs_bootstrapped = []
    aucs_bootstrapped = []

    # Instead, we specify a base fpr array, and interpolate the tpr values onto it.
    base_fpr = np.linspace(0, 1, 101)

    # Bootstrap TPRs
    for _ in range(n_bootstraps):
        y_resampled, y_hat_probs_resampled = resample(y, y_hat_probs)  # type: ignore
        fpr_resampled, tpr_resampled, _ = roc_curve(y_resampled, y_hat_probs_resampled)

        tpr_bootstrapped = np.interp(base_fpr, fpr_resampled, tpr_resampled)
        tpr_bootstrapped[0] = 0.0
        tprs_bootstrapped.append(tpr_bootstrapped)

        auc_boot = roc_auc_score(y_resampled, y_hat_probs_resampled)
        aucs_bootstrapped.append(auc_boot)

    return np.array(tprs_bootstrapped), np.array(aucs_bootstrapped), base_fpr


def plot_auc_roc(
    eval_dataset: EvalDataset,
    title: str = "Receiver Operating Characteristic (ROC) Curve",
    n_bootstraps: int = 100,
) -> pn.ggplot:
    """Plot AUC ROC curve with bootstrapped 95% confidence interval using Seaborn.

    Args:
        eval_dataset (EvalDataset): Evaluation dataset.
        title (str, optional): title. Defaults to "ROC-curve".
        fig_size (Optional[tuple], optional): figure size. Defaults to None.
        dpi (int, optional): dpi. Defaults to 160.
        save_path (Optional[Path], optional): path to save figure. Defaults to None.
        n_bootstraps (int, optional): number of bootstraps. Defaults to 1000.

    Returns:
        Union[None, Path]: None if save_path is None, else path to saved figure.
    """
    y = eval_dataset.y
    y_hat_probs = eval_dataset.y_hat_probs

    if isinstance(y, pd.DataFrame) or isinstance(y_hat_probs, pd.DataFrame):
        raise TypeError

    # We need a custom bootstrap implementation, because using scipy.bootstrap
    # on the roc_curve method will yield different fpr values for each resample,
    # and thus the tpr values will be interpolated on different fpr values. This
    # will result in arrays of different dimensions, which will cause an error.

    # Initialize lists for bootstrapped TPRs and FPRs
    tprs_bootstrapped, aucs_bootstrapped, base_fpr = bootstrap_roc(
        n_bootstraps=n_bootstraps,
        y=y,
        y_hat_probs=y_hat_probs,
    )

    mean_tprs = tprs_bootstrapped.mean(axis=0)
    se_tprs = tprs_bootstrapped.std(axis=0) / np.sqrt(n_bootstraps)

    # Calculate confidence interval for TPR over all FPRs
    tprs_upper = mean_tprs + se_tprs
    tprs_lower = mean_tprs - se_tprs

    # Calculate confidence interval for AUC
    auc_mean = np.mean(aucs_bootstrapped)
    auc_se = np.std(aucs_bootstrapped) / np.sqrt(n_bootstraps)
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
        label=f"AUROC (95% CI): {auc_mean:.2f} ({auc_ci[0]:.2f}-{auc_ci[1]:.2f})",
        x=1,
        y=0,
        ha="right",
        va="bottom",
        size=10,
    )

    # Plot AUC ROC curve
    return (
        pn.ggplot(df, pn.aes(x="fpr", y="tpr"))
        + pn.geom_line(size=1)
        + pn.geom_line(pn.aes(y="tpr_upper"), linetype="dashed", color="grey")
        + pn.geom_line(pn.aes(y="tpr_lower"), linetype="dashed", color="grey")
        + pn.labs(title=title, x="1 - Specificity", y="Sensitivity")
        + pn.xlim(0, 1)
        + pn.ylim(0, 1)
        + auroc_label
    )
