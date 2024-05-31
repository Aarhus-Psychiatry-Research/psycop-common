import plotnine as pn
import polars as pl


def plot(model: pl.DataFrame) -> pn.ggplot:
    model = _plot_transformations(model)

    plot = (
        pn.ggplot(model, pn.aes(x="Layer", y="auroc", ymin="lower", ymax="upper", color="Type"))
        + pn.geom_point(size=3)
        + pn.geom_errorbar(width=0.2)
        + pn.scale_y_continuous(limits=[0.75, 0.85], expand=(0, 0))
        + pn.labs(y="AUROC")
        + pn.theme(
            axis_title_x=pn.element_text(margin={"t": 20}),
            axis_text_x=pn.element_text(angle=45, hjust=1),
        )
    )
    return plot


def _plot_transformations(model: pl.DataFrame) -> pl.DataFrame:
    model = model.with_columns(
        pl.when(pl.col("run_name").str.contains("min"))
        .then(pl.lit("Aggregations"))
        .when(pl.col("run_name").str.contains("hparam"))
        .then(pl.lit("Lookbehinds"))
        .when(pl.col("run_name").str.contains("look"))
        .then(pl.lit("Lookbehinds"))
        .otherwise(pl.lit("Naive"))
        .alias("Type")
    ).with_columns(pl.col("run_name").str.extract(r"(\d).*").alias("Layer"))

    return model
