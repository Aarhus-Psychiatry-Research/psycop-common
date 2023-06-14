import pickle
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import plotnine as pn
import polars as pl
import shap
from care_ml.model_evaluation.config import (
    COLOURS,
    FIGURES_PATH,
    PN_THEME,
    TEXT_FIGURES_PATH,
)
from care_ml.model_evaluation.figures.feature_importance.shap.get_shap_values import (
    get_top_i_features_by_mean_abs_shap,
)
from care_ml.model_evaluation.utils.feature_name_to_readable import (
    feature_name_to_readable,
)
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap


def plot_shap_for_feature(
    df: pl.DataFrame,
    feature_name: str,
    model: Literal["baseline", "text"] = "baseline",
) -> pn.ggplot:
    # < 4 since a binary can have 0, 1 and null
    feature_is_binary = df["feature_value"].n_unique() < 4
    feature_is_categorical = df["feature_value"].n_unique() < 10

    if not feature_is_binary:
        df = df.filter(
            pl.col("feature_value") < pl.col("feature_value").quantile(0.995),
        )

    if feature_is_binary:
        df = df.with_columns(
            pl.when(pl.col("feature_value") == 1.0)
            .then(pl.lit("True"))
            .otherwise(pl.lit("False"))
            .keep_name(),
        )

    if feature_is_categorical:
        df = df.with_columns(pl.col("feature_value").cast(pl.Utf8).keep_name())

    p = (
        pn.ggplot(df, pn.aes(x="feature_value", y="shap_value"))
        + pn.geom_hline(yintercept=0, size=0.3, colour="grey")
        + pn.geom_point(
            alpha=0.05,
            shape="o",
            position="jitter",
            size=2,
            colour=COLOURS["blue"],
        )
        + pn.xlab(f"{feature_name}")
        + pn.ylab("SHAP")
        + pn.ggtitle(f"{model.capitalize()} model")
        + PN_THEME
    )

    return p


def plot_top_i_shap(
    shap_long_df: pl.DataFrame,
    i: int,
    model: Literal["baseline", "text"] = "baseline",
) -> dict[str, pn.ggplot]:
    df = get_top_i_features_by_mean_abs_shap(shap_long_df=shap_long_df, i=i)

    plots = {}

    for feature_rank in range(1, i + 1):
        feature_df = df.filter(
            pl.col("shap_mean_rank") == feature_rank,
        )
        feature_name = feature_name_to_readable(feature_df["feature_name"][0])
        p = plot_shap_for_feature(df=feature_df, feature_name=feature_name, model=model)
        plots[str(feature_rank)] = p

    return plots


def save_plots_for_top_i_shap_by_mean_abs(
    shap_long_df: pl.DataFrame,
    i: int,
    save_dir: Path,
    model: Literal["baseline", "text"] = "baseline",
) -> Path:
    plots = plot_top_i_shap(i=i, shap_long_df=shap_long_df, model=model)

    for feature_i, plot in plots.items():
        print(f"Plotting SHAP panel {feature_i}")
        plot.save(save_dir / f"plot_{feature_i}.jpg", dpi=600)

    return save_dir


def plot_shap_summary(
    shap_values: bytes,
    model: Literal["baseline", "text"],
    max_display: int = 20,
) -> None:
    """Plot summary/beeswarm plot of SHAP values

    Args:
        shap_values (bytes): SHAP values from explainer
        model: (Literal["baseline", "text"]). Model to get SHAP values from.
        max_display (int, optional): Number of features to display. Defaults to 20.
    """

    shap_values = pickle.loads(shap_values)

    # Generate colormap
    cmap = LinearSegmentedColormap.from_list(
        "",
        [COLOURS["blue"], COLOURS["purple"], COLOURS["red"]],
    )

    # Change font to Times New Roman
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = "Times New Roman"

    shap.summary_plot(
        shap_values,
        show=False,
        plot_size=(12, 5),
        max_display=max_display,
        cmap=cmap,
    )
    if model == "baseline":
        output_path = FIGURES_PATH / "shap" / "shap_summary.png"
    elif model == "text":
        output_path = TEXT_FIGURES_PATH / "shap" / "shap_summary.png"
    else:
        raise ValueError(f"model is {model}, but must be 'baseline' or 'text.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.title(f"{model.capitalize()} model: SHAP values", fontsize=16, x=0.3, y=1.05)
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()
