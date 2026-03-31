from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import numpy as np
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.common.model_training_v2.config.config_utils import PsycopConfig

import pandas as pd
import plotnine as pn

from psycop.common.model_evaluation.binary.global_performance.roc_auc import bootstrap_roc
from psycop.common.model_evaluation.confusion_matrix.confusion_matrix import (
    get_confusion_matrix_cells_from_df,
)

import pandas as pd
import plotnine as pn

from psycop.common.model_evaluation.confusion_matrix.confusion_matrix import ConfusionMatrix
from psycop.common.test_utils.str_to_df import str_to_df
from psycop.projects.t2d_extended.model_evaluation.config import T2D_PN_THEME


def eval_df_to_eval_dataset(cfg: PsycopConfig) -> EvalDataset: 
    # dataset = pd.read_parquet(cfg['trainer']['validation_data']['paths']).rename(columns={"prediction_time_uuid": "pred_time_uuid"})

    # dataset.rename(columns={"prediction_time_uuid": "pred_time_uuids"})

    eval_df = pd.read_parquet(f"{cfg['logger']['*']['disk']['run_path']}/eval_df.parquet")
    
    eval_df["ids"] = eval_df["pred_time_uuid"].apply(lambda s: s.split('-')[0])
    eval_df["pred_timestamps"] = eval_df["pred_time_uuid"].apply(lambda s: s.split('-')[1:4])

    return EvalDataset(
        ids=eval_df["ids"],
        y=eval_df["y"],
        y_hat_probs=eval_df["y_hat_prob"],
        pred_timestamps=eval_df["pred_timestamps"],
        pred_time_uuids=eval_df["pred_time_uuid"],
    )




def plot_auc_roc(
    eval_dataset: EvalDataset,
    title: str = "", #Receiver Operating Characteristic (ROC) Curve
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
    auc_ci = np.percentile(aucs_bootstrapped, [2.5, 97.5])

    df = pd.DataFrame(
        {"fpr": base_fpr, "tpr": mean_tprs, "tpr_lower": tprs_lower, "tpr_upper": tprs_upper}
    )

    auroc_label = pn.annotate(
        "text",
        label=f"AUROC (95% CI): {auc_mean:.3f} ({auc_ci[0]:.3f}-{auc_ci[1]:.3f})",
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
        + pn.geom_abline(intercept=0, slope=1, linetype="dotted")
        + pn.labs(title=title, x="1 - Specificity", y="Sensitivity")
        + pn.xlim(0, 1)
        + pn.ylim(0, 1)
        + T2D_PN_THEME
        + auroc_label
        + pn.theme(
        figure_size=(5, 5),
        dpi=300,
    )
    )


def auroc_plot(cfg: PsycopConfig) -> pn.ggplot:

    eval_dataset = eval_df_to_eval_dataset(cfg)

    p = plot_auc_roc(eval_dataset=eval_dataset)

    p.save(f"{cfg['logger']['*']['disk']['run_path']}/auc_roc.png")

    return p


def confusion_matrix_plot(cfg: PsycopConfig) -> pn.ggplot:
    eval_dataset = eval_df_to_eval_dataset(cfg)

    df = pd.DataFrame(
        {
            "true": eval_dataset.y,
            "pred": eval_dataset.get_predictions_for_positive_rate(0.03)[0],
        }
    )
    confusion_matrix = get_confusion_matrix_cells_from_df(df=df)

    p = plotnine_confusion_matrix(
        matrix=confusion_matrix,
        outcome_text=f"T2D within {int(cfg['trainer']['training_preprocessing_pipeline']['*']['sufficient_lookahead_filter']['n_days']/365)} years",
    )

    p.save(f"{cfg['logger']['*']['disk']['run_path']}/confusion_matrix_plot.png")

    return p


def plotnine_confusion_matrix(matrix: ConfusionMatrix, outcome_text: str) -> pn.ggplot:
    df = str_to_df(
        f"""true,pred,estimate
+,+,"{f'{matrix.true_positives:,}'}",
+,-,"{f'{matrix.false_negatives:,}'}",
-,+,"{f'{matrix.false_positives:,}'}",
-,-,"{f'{matrix.true_negatives:,}'}",
" ",+,"PPV:\n{round(matrix.ppv*100, 1)}%",
" ",-,"NPV:\n{round(matrix.npv*100,1)}%",
-," ","Spec:\n{round(matrix.specificity*100, 1)}%",
+," ","Sens:\n{round(matrix.sensitivity*100, 1)}%",
"""
    )

    """Create a confusion matrix and return a plotnine object."""
    df["true"] = pd.Categorical(df["true"], ["+", "-", " "])
    df["pred"] = pd.Categorical(df["pred"], ["+", "-", " "])

    p = (
        pn.ggplot(df, pn.aes(x="true", y="pred", fill="estimate"))
        + T2D_PN_THEME
        + pn.geom_tile(pn.aes(width=0.95, height=0.95), fill="gainsboro")
        + pn.geom_text(pn.aes(label="estimate"), size=18, color="black")
        + pn.theme(
            axis_line=pn.element_blank(),
            axis_ticks=pn.element_blank(),
            axis_text=pn.element_text(size=15, color="black"),
            panel_grid_major=pn.element_blank(),
            panel_grid_minor=pn.element_blank(),
            panel_background=pn.element_blank(),
            legend_position="none",
            figure_size=(5, 5),
            dpi=300,
        )
        + pn.scale_y_discrete(reverse=True)
        + pn.labs(x=f"Actual {outcome_text}", y=f"Predicted {outcome_text}")
        
    )

    return p



if __name__ == "__main__":

    y = 23

    run_path = (
        f"E:/shared_resources/t2d_extended/training/T2D-extended, temporal validation/"
        f"2013-01-01_2018-01-01_20{y}-01-01_20{y}-12-31"
    )

    cfg = PsycopConfig().from_disk(f"{run_path}/config.cfg")

    auroc = auroc_plot(cfg=cfg)

    confusion_matrix = confusion_matrix_plot(cfg=cfg)

    print("heya")

