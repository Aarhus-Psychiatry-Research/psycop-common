from pathlib import Path

import pandas as pd
import plotnine as pn
import polars as pl

from psycop.common.model_evaluation.binary.utils import sensitivity_by_group
from psycop.projects.restraint.feature_generation.modules.loaders.load_restraint_outcome_timestamps import (
    load_restraint_outcome_timestamps,
)
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.auroc_by_model import auroc_by_model
from psycop.projects.restraint.evaluation.utils import (
    parse_dw_ek_borger_from_uuid,
    parse_timestamp_from_uuid,
    read_eval_df_from_disk,
)
from psycop.common.model_training.training_output.dataclasses import (
    get_predictions_for_positive_rate,
)

def plotnine_sensitivity_by_outcome(df: pd.DataFrame, title: str = "Sensitivity by Outcome") -> pn.ggplot:
    df["proportion_of_n"] = df["n_in_bin"] / df["n_in_bin"].sum()

    p = (
        pn.ggplot(df, pn.aes(x="type", y="sensitivity"))
        + pn.geom_bar(pn.aes(fill="type"), stat="identity")
        + pn.labs(x="Restraint Type", y="Sensitivity", title=title)
        + pn.ylim(0, 0.8)
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
            # figure_size=(4, 5),
        )
        + pn.scale_x_discrete()
        + pn.scale_fill_manual(values=["#669BBC", "#A8C686", "#F3A712"])
    )

    if "ci_lower" in df.columns:
        p += pn.geom_point(size=1)
        p += pn.geom_errorbar(pn.aes(ymin="ci_lower", ymax="ci_upper"), width=0.1)

    return p

def plotnine_sensitivity_by_first_outcome(df: pd.DataFrame, title: str = "Sensitivity by First Outcome") -> pn.ggplot:
    df["proportion_of_n"] = df["n_in_bin"] / df["n_in_bin"].sum()

    p = (
        pn.ggplot(df, pn.aes(x="outcome", y="sensitivity"))
        + pn.geom_bar(pn.aes(fill="outcome"), stat="identity")
        + pn.labs(x="Restraint Type", y="Sensitivity", title=title)
        + pn.ylim(0, 0.8)
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
            # figure_size=(4, 5),
        )
        + pn.scale_x_discrete()
        + pn.scale_fill_manual(values=["#669BBC", "#A8C686", "#F3A712"])
    )

    if "ci_lower" in df.columns:
        p += pn.geom_point(size=1)
        p += pn.geom_errorbar(pn.aes(ymin="ci_lower", ymax="ci_upper"), width=0.1)

    return p

def sensitivity_by_restraint_type(df: pd.DataFrame) -> pd.DataFrame:
    mechanical = sensitivity_by_group(df, groupby_col_name="mechanical_restraint").iloc[1, 2:]
    mechanical.name = "Mechanical"

    chemical = sensitivity_by_group(df, groupby_col_name="chemical_restraint").iloc[1, 2:]
    chemical.name = "Chemical"

    manual = sensitivity_by_group(df, groupby_col_name="manual_restraint").iloc[1, 2:]
    manual.name = "Manual"
    
    all = pd.concat([mechanical, chemical, manual], axis=1).T

    all["type"] = all.index

    return all

def sensitivity_by_outcome_model(df: pl.DataFrame, outcome_df: pl.DataFrame) -> pd.DataFrame:
    positives = parse_timestamp_from_uuid(parse_dw_ek_borger_from_uuid(df)).filter(pl.col("y") == 1)
    
    outcome_df = outcome_df.with_columns(pl.when(pl.col("mechanical_restraint").is_not_null()).then(1).otherwise(0).alias("mechanical_restraint"))
    outcome_df = outcome_df.with_columns(pl.when(pl.col("chemical_restraint").is_not_null()).then(1).otherwise(0).alias("chemical_restraint"))
    outcome_df = outcome_df.with_columns(pl.when(pl.col("manual_restraint").is_not_null()).then(1).otherwise(0).alias("manual_restraint"))

    eval_df = (
        positives.join(outcome_df, on="dw_ek_borger", how="left")
    ).to_pandas()

    eval_df = eval_df[((eval_df["timestamp"].dt.date + pd.DateOffset(2)) >= eval_df["datotid_start"].dt.date) & (eval_df["timestamp"].dt.date <= eval_df["datotid_slut"].dt.date)] # type: ignore

    eval_df["y_hat"] = get_predictions_for_positive_rate(0.5, eval_df.y_hat_prob)[0]

    return sensitivity_by_restraint_type(eval_df)

def sensitivity_by_first_outcome_model(df: pl.DataFrame, outcome_df: pl.DataFrame) -> pd.DataFrame:
    positives = parse_timestamp_from_uuid(parse_dw_ek_borger_from_uuid(df)).filter(pl.col("y") == 1)
    
    eval_df = (
        positives.join(outcome_df, on="dw_ek_borger", how="left")
    ).to_pandas()

    eval_df = eval_df[((eval_df["timestamp"].dt.date + pd.DateOffset(2)) >= eval_df["datotid_start"].dt.date) & (eval_df["timestamp"].dt.date <= eval_df["datotid_slut"].dt.date)] # type: ignore

    eval_df["y_hat"] = get_predictions_for_positive_rate(0.5, eval_df.y_hat_prob)[0]

    eval_df["outcome"] = eval_df[["mechanical_restraint", "chemical_restraint", "manual_restraint"]].idxmin(axis=1).replace({"mechanical_restraint": "Mechanical", "chemical_restraint": "Chemical", "manual_restraint": "Manual"})

    return sensitivity_by_group(eval_df, groupby_col_name="outcome")

if __name__ == "__main__":
    save_dir = Path(__file__).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    df = read_eval_df_from_disk(
        "E:/shared_resources/restraint/eval_runs/restraint_all_tuning_best_run_evaluated_on_test"
    )
    outcome_df = pl.from_pandas(load_restraint_outcome_timestamps())

    plotnine_sensitivity_by_first_outcome(sensitivity_by_first_outcome_model(df=df, outcome_df=outcome_df)).save(
         save_dir / "restraint_sensitivity_by_first_outcome.png"
    )

    plotnine_sensitivity_by_outcome(sensitivity_by_outcome_model(df=df, outcome_df=outcome_df)).save(
         save_dir / "restraint_sensitivity_by_outcome.png"
    )


