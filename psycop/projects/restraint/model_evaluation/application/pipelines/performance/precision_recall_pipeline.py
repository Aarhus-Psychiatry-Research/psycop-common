from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as pn
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.utils import resample

from psycop.projects.restraint.model_evaluation.config import (
    EVAL_RUN,
    FIGURES_PATH,
    MODEL_NAME,
    PN_THEME,
    TEXT_EVAL_RUN,
    TEXT_FIGURES_PATH,
)
from psycop.projects.restraint.utils.best_runs import Run


def bootstrap_pr(
    n_bootstraps: int,
    y: pd.Series,
    y_hat_probs: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    recs_bootstrapped = []
    prec_bootstrapped = []

    # Instead, we specify a base fpr array, and interpolate the tpr values onto it.
    base_fpr = np.linspace(0, 1, 101)

    # Bootstrap TPRs
    for _ in range(n_bootstraps):
        y_resampled, y_hat_probs_resampled = resample(y, y_hat_probs)  # type: ignore
        prec_resampled, rec_resampled, _ = precision_recall_curve(
            y_resampled,
            y_hat_probs_resampled,
        )

        rec_bootstrapped = np.interp(base_fpr, prec_resampled, rec_resampled)
        rec_bootstrapped[0] = 0.0
        recs_bootstrapped.append(rec_bootstrapped)

        prec_boot = average_precision_score(y_resampled, y_hat_probs_resampled)
        prec_bootstrapped.append(prec_boot)

    return np.array(recs_bootstrapped), np.array(prec_bootstrapped), base_fpr


def precision_recall_pipeline(run: Run, path: Path):
    eval_ds = run.get_eval_dataset()

    if isinstance(eval_ds.y, pd.DataFrame) or isinstance(
        eval_ds.y_hat_probs,
        pd.DataFrame,
    ):
        raise TypeError

    recs_bootstrapped, prec_bootstrapped, base_fpr = bootstrap_pr(
        n_bootstraps=1000,
        y=eval_ds.y,
        y_hat_probs=eval_ds.y_hat_probs,
    )

    mean_recs = recs_bootstrapped.mean(axis=0)
    se_recs = recs_bootstrapped.std(axis=0) / np.sqrt(1000)

    # Calculate confidence interval for TPR over all FPRs
    recs_upper = mean_recs + se_recs
    recs_lower = mean_recs + se_recs

    # Calculate confidence interval for AUC
    prec_mean = np.mean(prec_bootstrapped)
    prec_se = np.std(prec_bootstrapped) / np.sqrt(1000)
    prec_ci = [prec_mean - 1.96 * prec_se, prec_mean + 1.96 * prec_se]

    auprc_label = pn.annotate(
        "text",
        label=f"AUPRC: {prec_mean:.3f} ({prec_ci[0]:.3f}-{prec_ci[1]:.3f})",
        x=1,
        y=1,
        ha="right",
        va="top",
        size=18,
    )

    df = pd.DataFrame(
        {
            "fpr": base_fpr,
            "rec": mean_recs,
            "rec_lower": recs_lower,
            "rec_upper": recs_upper,
        },
    )

    (
        pn.ggplot(df, pn.aes(x="fpr", y="rec"))
        + pn.geom_line(size=1)
        + pn.labs(
            title=f"AUPRC for {MODEL_NAME[run.name]}",
            x="Precision (positive predictive value)",
            y="Recall (sensitivity)",
        )
        + pn.xlim(0, 1)
        + pn.ylim(0, 1)
        + PN_THEME
        + pn.theme(
            axis_text=pn.element_text(size=15),
            axis_title=pn.element_text(size=18),
            plot_title=pn.element_text(size=22),
            legend_position="none",
        )
        + auprc_label
    ).save(path / "precision_recall.png", dpi=300)


if __name__ == "__main__":
    precision_recall_pipeline(EVAL_RUN, FIGURES_PATH)
    precision_recall_pipeline(TEXT_EVAL_RUN, TEXT_FIGURES_PATH)
