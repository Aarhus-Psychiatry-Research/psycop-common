"""Evaluate model performance by age.
Generates a bar plot showing the proportion of predictions made on each age group, with an overlaid line plot showing the AUROC for each age group.
"""

from collections.abc import Sequence

import pandas as pd
import plotnine as pn
import polars as pl

from psycop.common.feature_generation.loaders_2025.demographics import birthdays
from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.projects.t2d_extended.model_evaluation.config import T2D_PN_THEME
from psycop.projects.t2d_extended.model_evaluation.utils.parse_utils import (
    eval_df_to_eval_dataset,
    fix_pred_timestamps,
    parse_dw_ek_borger_from_uuid,
)
from psycop.projects.t2d_extended.model_evaluation.utils.auroc_utils import auroc_by_model


def add_age(df: pl.DataFrame, birthdays: pl.DataFrame, age_col_name: str = "age") -> pl.DataFrame:
    df = df.join(birthdays, on="dw_ek_borger", how="left")
    df = df.with_columns(
        ((pl.col("pred_timestamps") - pl.col("date_of_birth")).dt.total_days()).alias(age_col_name)
    )
    df = df.with_columns((pl.col(age_col_name) / 365.25).alias(age_col_name))

    return df


def auroc_by_age_model(
    df: pl.DataFrame, birthdays: pl.DataFrame, bins: Sequence[float]
) -> pd.DataFrame:
    eval_df = (
        add_age(fix_pred_timestamps(parse_dw_ek_borger_from_uuid(df)), birthdays)
    ).to_pandas()

    df = pl.DataFrame(
        auroc_by_model(
            input_values=eval_df["age"],
            y=eval_df["y"],
            y_hat_probs=eval_df["y_hat_probs"],
            input_name="age",
            bins=bins,
        )
    )

    return df.to_pandas()


def plotnine_auroc_by_age(df: pd.DataFrame, title: str = "") -> pn.ggplot:
    df = df.copy()
    df["proportion_of_n"] = df["n_in_bin"] / df["n_in_bin"].sum()
    df["auroc_label"] = df["auroc"].round(2).map(lambda x: f"{x:.2f}")

    if "ci_upper" in df.columns:
        df["auroc_text_y"] = (df["ci_upper"] + 0.03).clip(upper=0.98)
    else:
        df["auroc_text_y"] = (df["auroc"] + 0.03).clip(upper=0.98)

    p = (
        pn.ggplot(df, pn.aes(x="age_binned", y="auroc"))
        + pn.geom_bar(
            pn.aes(x="age_binned", y="proportion_of_n", fill="age_binned"), stat="identity"
        )
        + pn.geom_path(group=1, size=1)
        + pn.labs(x="Age in years", y="AUROC", title=title)
        + pn.geom_text(pn.aes(y="auroc_text_y", label="auroc_label"), size=9, va="bottom")
        + pn.ylim(0, 1)
        + pn.theme_bw()
        + T2D_PN_THEME
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
        + pn.scale_fill_manual(
            values=["#66A5BC", "#66A5BC", "#66A5BC", "#66A5BC", "#66A5BC", "#66A5BC"]
        )
        + pn.theme(legend_position="none")
    )

    if "ci_lower" in df.columns:
        p += pn.geom_point(size=1)
        p += pn.geom_errorbar(pn.aes(ymin="ci_lower", ymax="ci_upper"), width=0.1)

    return p


if __name__ == "__main__":
    y = 18

    training_start_date = ("2013-01-01",)
    training_end_date = ("2018-01-01",)
    evaluation_interval = (f"20{y}-01-01", f"20{y}-12-31")

    run_path = (
        f"E:/shared_resources/t2d_extended/training/T2D-extended, temporal validation/"
        f"2013-01-01_2018-01-01_20{y}-01-01_20{y}-12-31"
    )

    birthday_df = pl.from_pandas(birthdays())

    cfg = PsycopConfig().from_disk(f"{run_path}/config.cfg")

    eval_ds = eval_df_to_eval_dataset(cfg)

    model = auroc_by_age_model(
        df=eval_ds.to_polars(), birthdays=birthday_df, bins=[18, *range(20, 70, 10)]
    )

    plotnine_auroc_by_age(model).save(f"{run_path}/auroc_by_age.png")
