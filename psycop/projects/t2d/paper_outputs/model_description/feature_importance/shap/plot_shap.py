from pathlib import Path

import plotnine as pn
import polars as pl

from psycop.projects.t2d.paper_outputs.model_description.feature_importance.shap.get_shap_values import (
    get_top_i_features_by_mean_abs_shap,
)
from psycop.projects.t2d.utils.feature_name_to_readable import feature_name_to_readable


def plot_shap_for_feature(df: pl.DataFrame, feature_name: str) -> pn.ggplot:
    # < 4 since a binary can have 0, 1 and null
    feature_is_binary = df["feature_value"].n_unique() < 4
    feature_is_categorical = df["feature_value"].n_unique() < 10

    if not feature_is_binary:
        df = df.filter(pl.col("feature_value") < pl.col("feature_value").quantile(0.995))

    if feature_is_binary:
        df = df.with_columns(
            pl.when(pl.col("feature_value") == 1.0)
            .then(pl.lit("True"))
            .otherwise(pl.lit("False"))
            .keep_name()
        )

    if feature_is_categorical:
        df = df.with_columns(pl.col("feature_value").cast(pl.Utf8).keep_name())

    p = (
        pn.ggplot(df, pn.aes(x="feature_value", y="shap_value"))
        + pn.geom_point(alpha=0.05, shape="o", position="jitter", size=0.5)
        + pn.theme_bw()
        + pn.xlab(f"{feature_name}")
        + pn.ylab("SHAP")
    )

    return p


def plot_top_i_shap(shap_long_df: pl.DataFrame, i: int) -> dict[str, pn.ggplot]:
    df = get_top_i_features_by_mean_abs_shap(shap_long_df=shap_long_df, i=i)

    plots = {}

    for feature_rank in range(1, i + 1):
        feature_df = df.filter(pl.col("shap_std_rank") == feature_rank)
        feature_name = feature_name_to_readable(feature_df["feature_name"][0])
        p = plot_shap_for_feature(df=feature_df, feature_name=feature_name)
        plots[str(feature_rank)] = p

    return plots


def save_plots_for_top_i_shap_by_mean_abs(
    shap_long_df: pl.DataFrame, i: int, save_dir: Path
) -> Path:
    plots = plot_top_i_shap(i=i, shap_long_df=shap_long_df)

    for feature_i, plot in plots.items():
        print(f"Plotting SHAP panel {feature_i}")
        plot.save(save_dir / f"plot_{feature_i}.jpg", dpi=600)

    return save_dir
