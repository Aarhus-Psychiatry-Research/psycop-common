from pathlib import Path

import numpy as np
import polars as pl

from psycop.projects.t2d.paper_outputs.model_description.feature_importance.shap.plot_shap import (
    plot_shap_for_feature,
    save_plots_for_top_i_shap_by_mean_abs,
)

np.random.seed(42)


def test_shap_plot_aesthetics():
    n_rows = 1_000
    df = pl.DataFrame(
        {"feature_value": np.random.rand(n_rows), "multiplier": np.random.rand(n_rows)}
    )
    plot_df = df.with_columns((pl.col("feature_value") * pl.col("multiplier")).alias("shap_value"))

    plot_shap_for_feature(df=plot_df, feature_name="Test feature").draw()


def test_plot_top_i_shap(shap_long_df: pl.DataFrame, tmp_path: Path):
    save_plots_for_top_i_shap_by_mean_abs(shap_long_df=shap_long_df, i=3, save_dir=tmp_path)

    assert len(list(tmp_path.glob("*.jpg"))) == 3
