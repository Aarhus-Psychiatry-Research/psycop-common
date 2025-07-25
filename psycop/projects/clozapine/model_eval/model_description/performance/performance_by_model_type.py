import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.clozapine.model_eval.config import CLOZAPINE_EVAL_OUTPUT_DIR

if __name__ == "__main__":
    experiment_dicts = {
        "XGBoost - 365d_lookahead": {
            "365d_lookahead_Structured + TF-IDF": "clozapine hparam, structured_text, xgboost, no lookbehind filter",
            "365d_lookahead_Structured": "clozapine hparam, structured_only, xgboost, no lookbehind filter",
            "365d_lookahead_TF-IDF 180 days": "clozapine hparam, tfidf_180_days, xgboost, no lookbehind filter",
            "365d_lookahead_Only unique_count_antipsychotics": "clozapine hparam, only_unique_count_antipsychotics, xgboost, no lookbehind filter",
        },
        "XGBoost - 730d_lookahead": {
            "730d_lookahead_Structured + TF-IDF": "clozapine hparam, structured_text_lookahead_730_days, xgboost, no lookbehind filter",
            "730d_lookahead_Structured": "clozapine hparam, only_structured_lookahead_730_days, xgboost, no lookbehind filter",
            "730d_lookahead_Only unique_count_antipsychotics": "clozapine hparam, only_unique_count_antipsychotics, xgboost, no lookbehind filter",
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
    method_df.to_pandas().to_excel(
        CLOZAPINE_EVAL_OUTPUT_DIR / "performance_by_predictor_set_and_lookahead.xlsx"
    )
