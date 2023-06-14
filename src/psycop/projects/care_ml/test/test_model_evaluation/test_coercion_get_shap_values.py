import polars as pl

from care_ml.model_evaluation.figures.feature_importance.shap.get_shap_values import (
    get_top_i_features_by_mean_abs_shap,
)


def test_get_top_i_shap(shap_long_df: pl.DataFrame):
    df = get_top_i_features_by_mean_abs_shap(i=2, shap_long_df=shap_long_df)

    # Feature 2 has the largest standard deviation
    assert set(df["feature_name"].unique()) == {"feature_2", "feature_3"}
    assert {
        "feature_name",
        "feature_value",
        "pred_time_index",
        "shap_value",
        "shap_mean_rank",
    } == set(df.columns)
