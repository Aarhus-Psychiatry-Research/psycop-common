from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as pn
from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper

from psycop.common.model_evaluation.binary.global_performance.roc_auc import bootstrap_roc
from psycop.projects.cvd.model_evaluation.single_run.auroc.model import AUROC
from sklearn.metrics import roc_auc_score

def auroc_plot(data: AUROC, title: str = "AUROC") -> pn.ggplot:
    auroc_label = pn.annotate(
            "text",
            label=f"AUROC (95% CI): {data.mean:.2f} ({data.ci[0]:.2f}-{data.ci[1]:.2f})",
            x=1,
            y=0,
            ha="right",
            va="bottom",
            size=20,
        )
    
    p = (
            pn.ggplot(data.to_dataframe(), pn.aes(x="fpr", y="tpr"))
            + pn.geom_ribbon(pn.aes(x="fpr", ymin="tpr_lower", ymax="tpr_upper"), fill="#B7C8B5")
            + pn.geom_line(pn.aes(x="fpr", y="tpr"), size=0.5, color="black")
            # + pn.geom_line(pn.aes(y="tpr_upper"), linetype="dashed", color="grey")
            # + pn.geom_line(pn.aes(y="tpr_lower"), linetype="dashed", color="grey")
            + pn.labs(title=title, x="1 - Specificity", y="Sensitivity")
            + pn.xlim(0, 1)
            + pn.ylim(0, 1)
            + pn.geom_abline(intercept=0, slope=1, linetype="dotted")
            + auroc_label
            + pn.theme_minimal()
            + pn.theme(
                axis_ticks=pn.element_blank(),
                panel_grid_minor=pn.element_blank(),
                text=(pn.element_text(family="Times New Roman")),
                axis_text_x=pn.element_text(size=15),
                axis_text_y=pn.element_text(size=15),
                axis_title=pn.element_text(size=22),
                plot_title=pn.element_text(size=30, ha="center"),
                dpi=300,
            )
        )
    
    return p


def auroc_model(df: pd.DataFrame, n_bootstraps: int = 5) -> AUROC: 
    y = df["y"]
    y_hat_probs = df["y_hat_prob"]

    # We need a custom bootstrap implementation, because using scipy.bootstrap
    # on the roc_curve method will yield different fpr values for each resample,
    # and thus the tpr values will be interpolated on different fpr values. This
    # will result in arrays of different dimensions, which will cause an error.

    # Initialize lists for bootstrapped TPRs and FPRs
    tprs_bootstrapped, aucs_bootstrapped, base_fpr = bootstrap_roc(
        n_bootstraps=n_bootstraps, y=y, y_hat_probs=y_hat_probs
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
    
    return AUROC(mean=roc_auc_score(y_true=y, y_score=y_hat_probs),  # type: ignore
        ci=(auc_ci[0], auc_ci[1]),
        fpr=base_fpr,
        mean_tprs=mean_tprs,
        tprs_lower=tprs_lower,
        tprs_upper=tprs_upper,)


if __name__ == "__main__":
    save_dir = Path(__file__).parent / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_experiment = "restraint_text_hyper"
    eval_data = MlflowClientWrapper().get_best_run_from_experiment(
            experiment_name=best_experiment, metric="all_oof_BinaryAUROC"
        ).eval_frame().frame.to_pandas()
    
    auroc_plot(auroc_model(eval_data)).save(save_dir / "auroc.png")