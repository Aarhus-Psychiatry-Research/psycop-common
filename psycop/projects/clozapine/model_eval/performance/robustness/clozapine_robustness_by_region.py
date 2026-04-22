from pathlib import Path

import pandas as pd
import plotnine as pn
import polars as pl

from psycop.common.model_training_v2.trainer.preprocessing.steps.geographical_split._geographical_split import (
    add_shak_to_region_mapping,
    load_shak_to_location_mapping,
)
from psycop.projects.clozapine.feature_generation.cohort_definition.clozapine_cohort_definition import (
    ClozapineCohortDefiner,
)
from psycop.projects.clozapine.loaders.visits import physical_visits_clozapine_2024
from psycop.projects.clozapine.model_eval.utils import (
    parse_dw_ek_borger_from_uuid,
    parse_timestamp_from_uuid,
    read_eval_df_from_disk,
)
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.auroc_by_model import auroc_by_model


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
        + pn.labs(
            x="Region", y="AUROC", title=title
        )  # + pn.geom_text(position=pn.position_stack(vjust=1))
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
            panel_grid_minor=pn.element_blank(),  # text=(pn.element_text(family="Times New Roman")),
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
    pred_times_bundle = (
        ClozapineCohortDefiner.get_filtered_prediction_times_bundle().prediction_times.frame
    )
    pred_times = pl.DataFrame(pred_times_bundle).select(pl.col(["dw_ek_borger", "timestamp"]))

    shak_to_location_df = load_shak_to_location_mapping()

    visits = pl.from_pandas(
        physical_visits_clozapine_2024(shak_code=6600, return_shak_location=True)
    ).rename({"timestamp": "visit_timestamp"})

    sorted_all_visits_df = add_shak_to_region_mapping(
        visits=visits,
        shak_to_location_df=shak_to_location_df,
        shak_codes_to_drop=["6600310", "6600021"],
    ).sort(["dw_ek_borger", "visit_timestamp"])

    pred_times_with_region = pred_times.join(
        sorted_all_visits_df,
        left_on=["dw_ek_borger", "timestamp"],
        right_on=["dw_ek_borger", "visit_timestamp"],
        how="left",
    )

    eval_dataset = (
        df.pipe(parse_dw_ek_borger_from_uuid).pipe(
            parse_timestamp_from_uuid, output_col_name="timestamp"
        )
    ).with_columns(pl.col("timestamp").dt.cast_time_unit("ns"))

    eval_df = eval_dataset.join(
        pred_times_with_region,
        left_on=["dw_ek_borger", "timestamp"],
        right_on=["dw_ek_borger", "timestamp"],
        how="left",
    ).to_pandas()

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
    experiment_name = "clozapine hparam, structured_text_365d_lookahead, xgboost, 1 year lookbehind filter, 2025_random_split"
    best_pos_rate = 0.05
    eval_dir = (
        f"E:/shared_resources/clozapine/eval_runs/{experiment_name}_best_run_evaluated_on_test"
    )
    eval_df = read_eval_df_from_disk(eval_dir)

    save_dir = Path("E:/shared_resources/clozapine/eval/figures")

    plotnine_auroc_by_region(auroc_by_region_model(df=eval_df)).save(
        save_dir / "clozapine_auroc_by_region.png"
    )
