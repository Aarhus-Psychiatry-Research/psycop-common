from pathlib import Path

import pandas as pd
import plotnine as pn
import polars as pl

from psycop.common.model_evaluation.binary.time.periodic_data import roc_auc_by_periodic_time_df
from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.projects.t2d_extended.model_evaluation.config import T2D_PN_THEME
from psycop.projects.t2d_extended.model_training.USING_temporal_evaluation import eval_df_to_eval_dataset




def parse_dw_ek_borger_from_uuid(
    df: pl.DataFrame, output_col_name: str = "dw_ek_borger"
) -> pl.DataFrame:
    return df.with_columns(
        pl.col("pred_time_uuids").str.split("-").list.first().cast(pl.Int64).alias(output_col_name)
    )


def fix_pred_timestamps(
    df: pl.DataFrame,
    col: str = "pred_timestamps_joined",
    output_col: str = "pred_timestamps",
) -> pl.DataFrame:
    
    df = df.with_columns(
    pl.col("pred_timestamps").list.join("-").alias("pred_timestamps_joined")
    )

    return df.with_columns(
        pl.col(col)
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f", strict=False)
        .alias(output_col)
    )
def plotnine_auroc_by_month(
    df: pd.DataFrame, title: str = ""
) -> pn.ggplot:

    df = df.copy()

    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]

    month_abbrev = {
        "January": "Jan",
        "February": "Feb",
        "March": "Mar",
        "April": "Apr",
        "May": "May",
        "June": "Jun",
        "July": "Jul",
        "August": "Aug",
        "September": "Sep",
        "October": "Oct",
        "November": "Nov",
        "December": "Dec",
    }

    df["time_bin"] = pd.Categorical(
        df["time_bin"],
        categories=month_order,
        ordered=True,
    )

    df["time_bin"] = df["time_bin"].map(month_abbrev)


    df["proportion_of_n"] = df["n_in_bin"] / df["n_in_bin"].sum()
    df["auroc_label"] = df["auroc"].round(2).map(lambda x: f"{x:.2f}")

    if "ci_upper" in df.columns:
        df["auroc_text_y"] = (df["ci_upper"] + 0.03).clip(upper=0.98)
    else:
        df["auroc_text_y"] = (df["auroc"] + 0.03).clip(upper=0.98)

    p = (
        pn.ggplot(df, pn.aes(x="time_bin", y="auroc"))
        + pn.geom_bar(
            pn.aes(y="proportion_of_n", fill="time_bin"),
            stat="identity",
        )
        + pn.geom_path(group=1, size=1)

        + pn.geom_text(
            pn.aes(y="auroc_text_y", label="auroc_label"),
            size=9,
            va="bottom",
        )

        + pn.labs(
            x="Month",
            y="AUROC",
            title="",  # remove title
        )

        + pn.ylim(0, 1)
        + pn.scale_x_discrete()
        + pn.scale_fill_manual(values=["#66A5BC"] * 12)

        + pn.theme_minimal()
        + T2D_PN_THEME

        + pn.theme(
            axis_text_x=pn.element_text(
                size=15,
                rotation=90,   # vertical
                ha="center",
                va="top",
            ),
            axis_text_y=pn.element_text(size=15),

            legend_position="none",  # remove legend

            panel_grid_minor=pn.element_blank(),

            axis_title=pn.element_text(size=22),
            plot_title=pn.element_blank(),  # remove title fully

            dpi=300,
            figure_size=(5, 5),
        )
    )

    if "ci_lower" in df.columns:
        p += pn.geom_point(size=1)
        p += pn.geom_errorbar(
            pn.aes(ymin="ci_lower", ymax="ci_upper"),
            width=0.1,
        )

    return p



def auroc_by_month_model(df: pl.DataFrame) -> pd.DataFrame:
    eval_df = fix_pred_timestamps(df).to_pandas()

    df = pl.DataFrame(
        roc_auc_by_periodic_time_df(
            labels=eval_df["y"],
            y_hat_probs=eval_df["y_hat_probs"],
            timestamps=eval_df["pred_timestamps"],
            bin_period="M",
        )
    )

    return df.to_pandas()


if __name__ == "__main__":
    y = 23

    training_start_date="2013-01-01",
    training_end_date="2018-01-01",
    evaluation_interval=(f"20{y}-01-01", f"20{y}-12-31")

    run_path = (
        f"E:/shared_resources/t2d_extended/training/T2D-extended, temporal validation/"
        f"2013-01-01_2018-01-01_20{y}-01-01_20{y}-12-31"
    )

    cfg = PsycopConfig().from_disk(f"{run_path}/config.cfg")


    eval_ds = eval_df_to_eval_dataset(cfg)

    model = auroc_by_month_model(df=eval_ds.to_polars())

    plotnine_auroc_by_month(model).save(
        f"{run_path}/auroc_by_month.png"
    )
