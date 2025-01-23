import plotnine as pn
import polars as pl

from psycop.projects.t2d.paper_outputs.config import COLORS, THEME


def plot(model: pl.DataFrame) -> pn.ggplot:
    model = _split_out_info(model)
    dodge = 0.1
    legend_title = "Experiment"
    model = model.rename({"Type": legend_title})

    plot = (
        pn.ggplot(
            model,
            pn.aes(
                x="Predictor layer",
                y="auroc",
                ymin="lower",
                ymax="upper",
                color=legend_title,
                shape="Estimator",
            ),
        )
        + pn.scale_color_manual(COLORS.color_scale())
        + pn.geom_point(size=4, position=pn.position_dodge(width=dodge))
        + pn.geom_errorbar(width=0.3, position=pn.position_dodge(width=dodge))
        + pn.labs(y="AUROC")
        + THEME
    )
    return plot


def table(model: pl.DataFrame, decimals: int = 2) -> pl.DataFrame:
    model = _split_out_info(model)
    return model.with_columns(
        pl.concat_str(
            [
                pl.col("auroc").round(decimals),
                pl.lit(" ("),
                pl.col("lower").round(decimals),
                pl.lit("-"),
                pl.col("upper").round(decimals),
                pl.lit(")"),
            ]
        )
    ).select(["Estimator", "Predictor layer", "Type", "auroc"])


def _split_out_info(model: pl.DataFrame) -> pl.DataFrame:
    model = (
        model.with_columns(
            pl.when(pl.col("run_name").str.contains("min"))
            .then(pl.lit("Aggregations (min, mean, max)"))
            .when(pl.col("run_name").str.contains("h,"))
            .then(pl.lit("Hyperparameter tuned"))
            .when(pl.col("run_name").str.contains("look"))
            .then(pl.lit("Lookbehinds (90, 365, 730 days)"))
            .when(pl.col("run_name").str.contains("SCORE2"))
            .then(pl.lit("SCORE2-like"))
            .otherwise(pl.lit("Baseline (default parameters, 730 days, mean)"))
            .alias("Type"),
            pl.col("run_name").str.extract(r"(\d).*").alias("Predictor layer"),
        )
        .with_columns(
            pl.when(pl.col("run_name").str.contains("XGB"))
            .then(pl.lit("XGBoost"))
            .otherwise(pl.lit("Logistic Regression"))
            .alias("Estimator")
        )
        .with_columns(
            pl.when(pl.col("Predictor layer").str.contains("1"))
            .then(pl.col("Predictor layer"))
            .when(pl.col("run_name").str.contains("SCORE2"))
            .then(pl.lit("SCORE2"))
            .otherwise(pl.lit("+") + pl.col("Predictor layer"))
            .cast(pl.Enum(["1", "+2", "+3", "+4", "SCORE2"]))
            .alias("Predictor layer")
        )
    )

    return model
