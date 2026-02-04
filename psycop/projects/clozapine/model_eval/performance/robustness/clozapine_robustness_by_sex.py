from pathlib import Path

import pandas as pd
import plotnine as pn
import polars as pl

from psycop.projects.clozapine.loaders.demographics import sex_female
from psycop.projects.clozapine.model_eval.utils import (
    parse_dw_ek_borger_from_uuid,
    read_eval_df_from_disk,
)
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.auroc_by_model import auroc_by_model


def plotnine_auroc_by_sex(df: pd.DataFrame, title: str = "AUROC by Sex") -> pn.ggplot:
    df["proportion_of_n"] = df["n_in_bin"] / df["n_in_bin"].sum()

    p = (
        pn.ggplot(df, pn.aes(x="sex", y="auroc"))
        + pn.geom_bar(pn.aes(x="sex", y="proportion_of_n", fill="sex"), stat="identity")
        + pn.geom_path(group=1, size=1)
        + pn.labs(x="Sex", y="AUROC", title=title)
        + pn.ylim(0, 1)
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(size=15),
            axis_text_y=pn.element_text(size=15),
            panel_grid_minor=pn.element_blank(),  # text=(pn.element_text(family="Times New Roman")),
            legend_position="none",
            axis_title=pn.element_text(size=22),
            plot_title=pn.element_text(size=30, ha="center"),
            dpi=300,
            figure_size=(5, 5),
        )
        + pn.scale_x_discrete()
        + pn.scale_fill_manual(values=["#669BBC", "#669BBC"])
    )

    if "ci_lower" in df.columns:
        p += pn.geom_point(size=1)
        p += pn.geom_errorbar(pn.aes(ymin="ci_lower", ymax="ci_upper"), width=0.1)

    return p


def auroc_by_sex_model(df: pl.DataFrame, sex_df: pl.DataFrame) -> pd.DataFrame:
    eval_df = (
        parse_dw_ek_borger_from_uuid(df).join(sex_df, on="dw_ek_borger", how="left")
    ).to_pandas()

    df = pl.DataFrame(
        auroc_by_model(
            input_values=eval_df["sex_female"],
            y=eval_df["y"],
            y_hat_probs=eval_df["y_hat_prob"],
            input_name="sex",
            bin_continuous_input=False,
        )
    )

    return df.with_columns(
        pl.when(pl.col("sex")).then(pl.lit("Female")).otherwise(pl.lit("Male")).alias("sex")
    ).to_pandas()


if __name__ == "__main__":
    experiment_name = "clozapine hparam, structured_text_365d_lookahead, xgboost, 1 year lookbehind filter, 2025_random_split"
    best_pos_rate = 0.05
    eval_dir = (
        f"E:/shared_resources/clozapine/eval_runs/{experiment_name}_best_run_evaluated_on_test"
    )
    eval_df = read_eval_df_from_disk(eval_dir)
    sex_df = pl.from_pandas(sex_female())

    save_dir = Path("E:/shared_resources/clozapine/eval/figures")

    plotnine_auroc_by_sex(auroc_by_sex_model(df=eval_df, sex_df=sex_df)).save(
        save_dir / "clozapine_auroc_by_sex_xgboost.png"
    )
