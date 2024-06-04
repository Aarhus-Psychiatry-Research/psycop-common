from typing import Literal

import pandas as pd
import plotnine as pn
import polars as pl
from sklearn.metrics import roc_auc_score, roc_curve

from psycop.projects.scz_bp.evaluation.scz_bp_run_evaluation_suite import (
    scz_bp_get_eval_ds_from_best_run_in_experiment,
)


def get_auc_roc_df(y: pd.Series, y_hat_probs: pd.Series) -> pl.DataFrame:  # type: ignore
    fpr, tpr, _ = roc_curve(y_true=y, y_score=y_hat_probs)
    return pl.DataFrame({"fpr": fpr, "tpr": tpr})


def scz_bp_auroc_by_data_type(
    modality2experiment_mapping: dict[str, str], model_type: Literal["joint", "scz", "bp"]
) -> pl.DataFrame:
    auc_roc_dfs = []
    aucs: dict[str, float] = {}
    for modality, experiment_name in modality2experiment_mapping.items():
        eval_df = scz_bp_get_eval_ds_from_best_run_in_experiment(
            experiment_name=experiment_name, model_type=model_type
        )
        auc_roc_df = get_auc_roc_df(y=eval_df.y, y_hat_probs=eval_df.y_hat_probs).with_columns(  # type: ignore
            pl.lit(modality).alias("modality")
        )  # type: ignore
        aucs[modality] = roc_auc_score(y_true=eval_df.y, y_score=eval_df.y_hat_probs)  # type: ignore
        auc_roc_dfs.append(auc_roc_df)

    auc_df = pl.DataFrame(aucs).melt(variable_name="modality", value_name="AUC")
    auc_roc_df = pl.concat(auc_roc_dfs)
    auc_roc_df = auc_roc_df.join(auc_df, how="left", on="modality")
    return auc_roc_df


def scz_bp_make_group_auc_plot(roc_df: pl.DataFrame) -> pn.ggplot:
    abline_df = pl.DataFrame({"fpr": [0, 1], "tpr": [0, 1]})

    roc_df = roc_df.with_columns(
        pl.concat_str(pl.col("modality"), pl.col("AUC").round(3), separator=": AUROC=")
    )
    order = (
        roc_df.group_by("modality")
        .agg(pl.col("AUC").max())
        .sort(by="AUC", descending=True)
        .get_column("modality")
        .to_list()
    )
    roc_df = roc_df.with_columns(pl.col("modality").cast(pl.Enum(order)))

    return (
        pn.ggplot(roc_df, pn.aes(x="fpr", y="tpr", color="modality"))
        + pn.geom_line()
        + pn.geom_line(data=abline_df, linetype="dashed", color="grey", alpha=0.5)
        + pn.labs(x="1 - Specificity", y="Sensitivty")
        + pn.coord_cartesian(xlim=(0, 1), ylim=(0, 1))
        + pn.theme_minimal()
        + pn.theme(
            legend_position=(0.65, 0.25),
            legend_direction="vertical",
            legend_title=pn.element_blank(),
            axis_title=pn.element_text(size=14),
            legend_text=pn.element_text(size=11),
            axis_text=pn.element_text(size=10),
            figure_size=(5, 5),
        )
    )


def plot_scz_bp_auroc_by_data_type(
    modality2experiment: dict[str, str], model_type: Literal["joint", "scz", "bp"]
) -> pn.ggplot:
    df = scz_bp_auroc_by_data_type(
        modality2experiment_mapping=modality2experiment, model_type=model_type
    )
    return scz_bp_make_group_auc_plot(df)


if __name__ == "__main__":
    modality2experiment = {
        "Structured + text": "sczbp/structured_text",
        "Structured only ": "sczbp/structured_only",
        "Text only": "sczbp/text_only",
    }
    df = scz_bp_auroc_by_data_type(
        modality2experiment_mapping=modality2experiment, model_type="joint"
    )
    p = scz_bp_make_group_auc_plot(df)
    p + pn.theme(
        legend_position=(0.65, 0.25),
        legend_direction="vertical",
        legend_title=pn.element_blank(),
        axis_title=pn.element_text(size=14),
        legend_text=pn.element_text(size=11),
        axis_text=pn.element_text(size=10),
        figure_size=(5, 5),
    )
