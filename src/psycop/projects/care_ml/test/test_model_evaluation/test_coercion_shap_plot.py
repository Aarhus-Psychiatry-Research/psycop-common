import numpy as np
import polars as pl

from care_ml.model_evaluation.figures.feature_importance.shap.shap_plots import (
    plot_shap_for_feature,
)

np.random.seed(42)


def test_shap_plot_aesthetics():
    n_rows = 1_000
    df = pl.DataFrame(
        {"feature_value": np.random.rand(n_rows), "multiplier": np.random.rand(n_rows)},
    )
    plot_df = df.with_columns(
        (pl.col("feature_value") * pl.col("multiplier")).alias("shap_value"),
    )

    plot_shap_for_feature(df=plot_df, feature_name="Test feature").draw()
