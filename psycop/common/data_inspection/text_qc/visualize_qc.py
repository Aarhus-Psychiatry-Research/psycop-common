from pathlib import Path

import plotnine as pn
import polars as pl


def plot_qc_results(df: pl.DataFrame) -> pn.ggplot:
    p = (
        pn.ggplot(df, pn.aes(x="overskrift", y="prop_true"))
        + pn.geom_col()
        + pn.theme(axis_text_x=pn.element_text(rotation=90))
    )
    return p


if __name__ == "__main__":
    cur_dir = Path(__file__).parent

    df = pl.read_csv(cur_dir / "log.csv", separator=";")
    df = (
        df.groupby("overskrift")
        .agg(pl.count(), pl.sum("useful").alias("n_true"))
        .with_columns((pl.col("n_true") / pl.col("count")).alias("prop_true"))
        .sort("prop_true", descending=True)
        .with_columns([pl.col("overskrift").cast(pl.Categorical)])
    )

    p = plot_qc_results(df)
    p.save(cur_dir / "qc_plot.png")
