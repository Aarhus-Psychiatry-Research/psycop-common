import polars as pl
from care_ml.model_evaluation.figures.feature_importance.shap.get_shap_values import (
    get_top_i_features_by_mean_abs_shap,
)
from care_ml.utils.feature_name_to_readable import feature_name_to_readable


def get_top_i_shap_values_for_printing(
    shap_long_df: pl.DataFrame,
    i: int,
) -> pl.DataFrame:
    aggregated = (
        get_top_i_features_by_mean_abs_shap(shap_long_df=shap_long_df, i=i)
        .groupby("feature_name")
        .agg(
            pl.col("shap_value").abs().mean().alias("Mean absolute SHAP"),
            pl.col("feature_name").first().alias("Feature"),
        )
    )

    ranked = aggregated.sort(by="Mean absolute SHAP", descending=True).select(
        pl.col("Mean absolute SHAP")
        .rank(method="average", descending=True)
        .cast(pl.Int64)
        .alias("Rank"),
        pl.col("Feature"),
        pl.col("Mean absolute SHAP").round(2).alias("Mean absolute SHAP"),
    )

    return ranked.with_columns(
        pl.col("Feature").apply(lambda x: feature_name_to_readable(x)).alias("Feature"),  # type: ignore
    )
