from pathlib import Path

import pandas as pd
import plotnine as pn
import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_evaluation.binary.time.periodic_data import roc_auc_by_periodic_time_df
from psycop.projects.restraint.evaluation.evaluation_utils import parse_timestamp_from_uuid


def plotnine_auroc_by_week(df: pd.DataFrame, title: str = "AUROC by Day of the Week") -> pn.ggplot:
    df["proportion_of_n"] = df["n_in_bin"] / df["n_in_bin"].sum()

    p = (
        pn.ggplot(df, pn.aes(x="time_bin", y="auroc"))
        + pn.geom_bar(pn.aes(x="time_bin", y="proportion_of_n", fill="time_bin"), stat="identity")
        + pn.geom_path(group=1, size=1)
        + pn.labs(x="Day", y="AUROC", title=title)
        + pn.ylim(0, 1)
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(size=15, rotation=45),
            axis_text_y=pn.element_text(size=15),
            panel_grid_minor=pn.element_blank(),
            text=(pn.element_text(family="Times New Roman")),
            legend_position="none",
            axis_title=pn.element_text(size=22),
            plot_title=pn.element_text(size=30, ha="center"),
            dpi=300,
        )
        + pn.scale_x_discrete()
        + pn.scale_fill_manual(
            values=[
                "#669BBC",
                "#A8C686",
                "#669BBC",
                "#A8C686",
                "#669BBC",
                "#A8C686",
                "#669BBC",
                "#A8C686",
                "#669BBC",
                "#A8C686",
                "#669BBC",
                "#A8C686",
            ]
        )
    )

    if "ci_lower" in df.columns:
        p += pn.geom_point(size=1)
        p += pn.geom_errorbar(pn.aes(ymin="ci_lower", ymax="ci_upper"), width=0.1)

    return p


def auroc_by_week(df: pl.DataFrame) -> pd.DataFrame:
    eval_df = parse_timestamp_from_uuid(df).to_pandas()

    df = pl.DataFrame(
        roc_auc_by_periodic_time_df(
            labels=eval_df["y"],
            y_hat_probs=eval_df["y_hat_prob"],
            timestamps=eval_df["timestamp"],
            bin_period="D",
        )
    )

    return df.to_pandas()


if __name__ == "__main__":
    save_dir = Path(__file__).parent / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_experiment = "restraint_text_hyper"
    best_pos_rate = 0.05
    df = (
        MlflowClientWrapper()
        .get_best_run_from_experiment(experiment_name=best_experiment, metric="all_oof_BinaryAUROC")
        .eval_frame()
        .frame
    )

    plotnine_auroc_by_week(auroc_by_week(df=df)).save(save_dir / "auroc_by_weekday.png")
