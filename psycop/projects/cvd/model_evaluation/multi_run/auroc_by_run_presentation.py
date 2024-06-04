import plotnine as pn
import polars as pl
from matplotlib import legend

from psycop.projects.t2d.paper_outputs.config import COLORS, THEME, Colors


def plot(model: pl.DataFrame) -> pn.ggplot:
    model = _plot_transformations(model)
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
                shape=legend_title,
            ),
        )
        + pn.scale_color_manual(COLORS.color_scale())
        + pn.geom_point(size=4, position=pn.position_dodge(width=dodge))
        + pn.geom_errorbar(width=0.3, position=pn.position_dodge(width=dodge))
        + pn.labs(y="AUROC")
        + THEME
    )
    return plot


def _plot_transformations(model: pl.DataFrame) -> pl.DataFrame:
    model = model.with_columns(
        pl.when(pl.col("run_name").str.contains("min"))
        .then(pl.lit("Aggregations (min, mean, max)"))
        .when(pl.col("run_name").str.contains("hparam"))
        .then(pl.lit("Hyperparameter tuned"))
        .when(pl.col("run_name").str.contains("look"))
        .then(pl.lit("Lookbehinds (90, 365, 730 days)"))
        .otherwise(pl.lit("Baseline (default parameters, 730 days, mean)"))
        .alias("Type")
    ).with_columns(pl.col("run_name").str.extract(r"(\d).*").alias("Predictor layer"))

    return model
