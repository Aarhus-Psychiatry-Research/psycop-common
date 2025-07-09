from pathlib import Path

import pandas as pd
import plotnine as pn
import polars as pl

from psycop.common.feature_generation.loaders.raw.load_visits import physical_visits
from psycop.common.model_training_v2.trainer.preprocessing.steps.geographical_split._geographical_split import (
    add_shak_to_region_mapping,
    load_shak_to_location_mapping,
)
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.auroc_by_model import auroc_by_model
from psycop.projects.restraint.evaluation.utils import (
    parse_dw_ek_borger_from_uuid,
    parse_timestamp_from_uuid,
    read_eval_df_from_disk,
)
from psycop.projects.restraint.feature_generation.modules.loaders.load_restraint_prediction_timestamps import (
    load_restraint_prediction_timestamps,
)


def plotnine_auroc_by_region(df: pd.DataFrame, title: str = "AUROC by Region") -> pn.ggplot:
    df["proportion_of_n"] = df["n_in_bin"] / df["n_in_bin"].sum()
    df["percentage_of_n"] = df["proportion_of_n"] * 100
    df["region_en"] = pd.Categorical(
        df["region"], categories=["West", "Central", "East"], ordered=True
    )
    df = df.sort_values("region_en")

    p = (
        pn.ggplot(df, pn.aes(x="region_en", y="auroc"))
        + pn.geom_bar(pn.aes(x="region_en", y="proportion_of_n", fill="region_en"), stat="identity")
        + pn.geom_path(group=1, size=1)
        + pn.labs(x="Region", y="AUROC", title=title)
        # + pn.geom_text(position=pn.position_stack(vjust=1))
        + pn.geom_text(
            pn.aes(x="region_en", y="proportion_of_n", fill="region_en", label="percentage_of_n"),
            va="bottom",
            format_string="{:.1f}%",
        )
        + pn.geom_text(
            pn.aes(x="region_en", y="auroc", label="auroc"),
            nudge_y=0.15,
            va="top",
            format_string="{:.2f}",
        )
        + pn.ylim(0, 1.1)
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(size=15),
            axis_text_y=pn.element_text(size=15),
            panel_grid_minor=pn.element_blank(),
            # text=(pn.element_text(family="Times New Roman")),
            legend_position="none",
            axis_title=pn.element_text(size=22),
            plot_title=pn.element_text(size=30, ha="center"),
            dpi=300,
        )
        + pn.scale_x_discrete()
        + pn.scale_fill_manual(values=["#669BBC", "#A8C686", "#F3A712"])
    )

    if "ci_lower" in df.columns:
        p += pn.geom_point(size=1)
        p += pn.geom_errorbar(pn.aes(ymin="ci_lower", ymax="ci_upper"), width=0.1)

    return p


def auroc_by_region_model(df: pl.DataFrame) -> pd.DataFrame:
    pred_times = pl.DataFrame(load_restraint_prediction_timestamps()).select(
        pl.col(["dw_ek_borger", "timestamp", "timestamp_discharge"])
    )

    df_with_admission_timestamp = (
        parse_timestamp_from_uuid(parse_dw_ek_borger_from_uuid(df))
        .with_columns(pl.col("timestamp").dt.cast_time_unit("ns"))
        .join(pred_times, on=["dw_ek_borger", "timestamp"], how="left")
    )

    shak_to_location_df = load_shak_to_location_mapping()

    visits = pl.from_pandas(physical_visits(shak_code=6600, return_shak_location=True))

    sorted_all_visits_df = add_shak_to_region_mapping(
        visits=visits,
        shak_to_location_df=shak_to_location_df,
        shak_codes_to_drop=["6600310", "6600021"],
    ).sort(["dw_ek_borger", "timestamp"])

    eval_df = (
        df_with_admission_timestamp.join(
            sorted_all_visits_df,
            left_on=["dw_ek_borger", "timestamp_discharge"],
            right_on=["dw_ek_borger", "timestamp"],
            how="left",
        )
    ).to_pandas()

    eval_df = eval_df[eval_df.region.notna()]

    eval_df = eval_df[eval_df.region != "na"]

    df = pl.DataFrame(
        auroc_by_model(
            input_values=eval_df["region"],
            y=eval_df["y"],
            y_hat_probs=eval_df["y_hat_prob"],
            input_name="region",
            bin_continuous_input=False,
        )
    )

    df = df.with_columns(
        pl.when(pl.col("region") == "vest")
        .then(pl.lit("West"))
        .when(pl.col("region") == "midt")
        .then(pl.lit("Central"))
        .otherwise(pl.lit("East"))
        .alias("region")
    )

    return df.to_pandas()


if __name__ == "__main__":
    save_dir = Path(__file__).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    best_experiment = "restraint_text_hyper"
    best_pos_rate = 0.05
    df = read_eval_df_from_disk(
        "E:/shared_resources/restraint/eval_runs/restraint_all_tuning_best_run_evaluated_on_test"
    )
    # region_df = _get_regional_split_df().collect()

    plotnine_auroc_by_region(auroc_by_region_model(df=df)).save(
        save_dir / "restraint_auroc_by_region.png"
    )
