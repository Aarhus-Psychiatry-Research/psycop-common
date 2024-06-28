from pathlib import Path
from typing import Sequence

import pandas as pd
import plotnine as pn
import polars as pl

from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays
from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.auroc_by_model import auroc_by_model
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.auroc_by_view import auroc_by_view
from psycop.projects.restraint.evaluation.evaluation_utils import add_age, parse_dw_ek_borger_from_uuid, parse_timestamp_from_uuid


def plotnine_auroc_by_age(df: pd.DataFrame, title: str = "AUROC by Age") -> pn.ggplot:
    df["proportion_of_n"] = df["n_in_bin"] / df["n_in_bin"].sum()

    p = (
        pn.ggplot(df, pn.aes(x="age_binned", y="auroc"))
        + pn.geom_bar(
            pn.aes(x="age_binned", y="proportion_of_n", fill="age_binned"), stat="identity"
        )
        + pn.geom_path(group=1, size=1)
        + pn.labs(x="Age", y="AUROC", title=title)
        + pn.ylim(0, 1)
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(size=15),
            axis_text_y=pn.element_text(size=15),
            panel_grid_minor=pn.element_blank(),
            text=(pn.element_text(family="Times New Roman")),
            legend_position="none",
            axis_title=pn.element_text(size=22),
            plot_title=pn.element_text(size=30, ha="center"),
            dpi=300,
        )
        + pn.scale_x_discrete()
        + pn.geom_text(pn.aes(label="n_in_bin"), nudge_y=-0.8, size=17, fontweight="bold", color="white")
        + pn.scale_fill_manual(values=["#669BBC", "#A8C686", "#669BBC", "#A8C686", "#669BBC", "#A8C686", "#669BBC", "#A8C686", "#669BBC", "#A8C686"])
    )

    if "ci_lower" in df.columns:
        p += pn.geom_point(size=1)
        p += pn.geom_errorbar(pn.aes(ymin="ci_lower", ymax="ci_upper"), width=0.1)

    return p


def auroc_by_age_model(df: pl.DataFrame, birthdays: pl.DataFrame, bins: Sequence[float]) -> pd.DataFrame:
    eval_df = (
        add_age(parse_timestamp_from_uuid(parse_dw_ek_borger_from_uuid(df)), birthdays)
    ).to_pandas()

    df = pl.DataFrame(auroc_by_model(
        input_values=eval_df["age"],
        y=eval_df["y"],
        y_hat_probs=eval_df["y_hat_prob"],
        input_name="age",
        bins=bins,
    ))

    return df.to_pandas()



if __name__ == "__main__":
    save_dir = Path(__file__).parent / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_experiment = "restraint_text_hyper"
    best_pos_rate = 0.05
    df = MlflowClientWrapper().get_best_run_from_experiment(
            experiment_name=best_experiment, metric="all_oof_BinaryAUROC"
        ).eval_frame().frame
    birthdays=pl.from_pandas(birthdays())

    plotnine_auroc_by_age(
            auroc_by_age_model(df=df, birthdays=birthdays, bins=[18, *range(20, 80, 10)])).save(save_dir / "auroc_by_age.png")
    
