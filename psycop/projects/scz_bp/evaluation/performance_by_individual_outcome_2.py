import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.scz_bp.evaluation.configs import SCZ_BP_EVAL_OUTPUT_DIR


def calculate_pretty_oof_performance(metric_frame: pl.DataFrame) -> pl.DataFrame:
    only_oof = (
        (
            metric_frame.filter(pl.col("metric").str.contains("out_of_fold"))
            .with_columns(pl.col("value").cast(pl.Float64))
            .select(
                pl.col("value").mean().round(3).alias("mean"),
                pl.col("value").std().round(3).alias("std"),
            )
        )
        .with_columns(
            pl.concat_str([pl.col("mean"), pl.lit(" ±"), pl.col("std")], separator="").alias(
                "pretty_value"
            )
        )
        .select("pretty_value")
        .rename({"pretty_value": "value"})
    )
    return only_oof


if __name__ == "__main__":
    client = MlflowClientWrapper()

    performance_dfs: list[pl.DataFrame] = []
    for diagnosis in ["scz", "bp"]:
        experiment_dicts = {
            "Train": {
                "Structured + text + synthetic": f"sczbp/{diagnosis}_ddpm",
                "Structured + text": f"sczbp/{diagnosis}_structured_text_xgboost",
                "Structured only": f"sczbp/{diagnosis}_structured_only-xgboost",
                "Text only": f"sczbp/{diagnosis}_tfidf_1000-xgboost",
            },
            "Test": {
                "Structured + text + synthetic": f"sczbp/test_{diagnosis}_structured_text_ddpm",
                "Structured + text": f"sczbp/test_{diagnosis}_structured_text",
                "Structured only": f"sczbp/test_{diagnosis}_structured_only",
                "Text only": f"sczbp/test_{diagnosis}_tfidf_1000",
            },
        }

        for split, experiment_dict in experiment_dicts.items():
            main_metric = "all_oof_BinaryAUROC" if split == "Train" else "BinaryAUROC"
            for feature_set, experiment_name in experiment_dict.items():
                metric_frame = (
                    client.get_best_run_from_experiment(
                        experiment_name=experiment_name, metric=main_metric
                    )
                    .get_all_metrics()
                    .melt(variable_name="metric")
                )
                if split == "Train":
                    performance_df = calculate_pretty_oof_performance(metric_frame=metric_frame)
                else:
                    performance_df = metric_frame.filter(
                        pl.col("metric") == main_metric
                    ).with_columns(pl.col("value").cast(pl.Float64).round(2).cast(pl.String))
                performance_dfs.append(
                    performance_df.with_columns(
                        pl.lit(split).alias("split"),
                        pl.lit(feature_set).alias("feature_set"),
                        pl.lit(diagnosis).alias("diagnosis"),
                    ).drop("metric")
                )

        performance_df = (
            pl.concat(performance_dfs, how="vertical")
            .pivot(values="value", index=["diagnosis", "feature_set"], columns="split")
            .with_columns(
                pl.col("feature_set").cast(
                    pl.Enum(
                        [
                            "Structured + text + synthetic",
                            "Structured + text",
                            "Text only",
                            "Structured only",
                        ]
                    )
                ),
                pl.col("diagnosis").cast(pl.Enum(["scz", "bp"])),
            )
            .sort("diagnosis", "feature_set")
        )
        performance_df.to_pandas().to_html(
            SCZ_BP_EVAL_OUTPUT_DIR / "performance_by_individual_outcome.html"
        )
