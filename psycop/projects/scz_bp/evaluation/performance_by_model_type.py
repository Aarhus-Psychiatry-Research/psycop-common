import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.scz_bp.evaluation.configs import SCZ_BP_EVAL_OUTPUT_DIR

if __name__ == "__main__":
    experiment_dicts = {
        "XGBoost": {
            "Structured only": "sczbp/structured_only-xgboost",
            "Text only": "sczbp/tfidf_1000-xgboost",
            "Structured + text": "sczbp/structured_text_xgboost",
        },
        "Logistic regression": {
            "Structured only": "sczbp/structured_only-logreg",
            "Text only": "sczbp/tfidf_1000-logreg",
            "Structured + text": "sczbp/structured_text_logreg",
        },
    }
    metrics_of_interest = ["all_oof_BinaryAUROC"]

    client = MlflowClientWrapper()

    method_dfs: list[pl.DataFrame] = []
    for model, experiment_dict in experiment_dicts.items():
        for feature_set_name, experiment_name in experiment_dict.items():
            metric_frame = (
                client.get_best_run_from_experiment(
                    experiment_name=experiment_name, metric="all_oof_BinaryAUROC"
                )
                .get_all_metrics()
                .melt(variable_name="metric")
            )

            only_oof = (
                (
                    metric_frame.filter(pl.col("metric").str.contains("out_of_fold"))
                    .with_columns(
                        pl.lit(feature_set_name).alias("feature_set"),
                        pl.col("value").cast(pl.Float64),
                    )
                    .group_by("feature_set")
                    .agg(
                        pl.col("value").mean().round(3).alias("mean"),
                        pl.col("value").std().round(3).alias("std"),
                    )
                )
                .with_columns(
                    pl.concat_str(
                        [pl.col("mean"), pl.lit(" ±"), pl.col("std")], separator=""
                    ).alias("pretty_value")
                )
                .select("feature_set", "pretty_value")
            ).with_columns(pl.lit(model).alias("model"))

            method_dfs.append(only_oof)

    method_df = pl.concat(method_dfs, how="vertical").pivot(
        values="pretty_value", index="model", columns="feature_set"
    )
    method_df.to_pandas().to_html(SCZ_BP_EVAL_OUTPUT_DIR / "performance_by_model_type.html")
