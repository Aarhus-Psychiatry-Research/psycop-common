import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.scz_bp.evaluation.configs import SCZ_BP_EVAL_OUTPUT_DIR


def calculate_pretty_oof_performance(
    metric_frame: pl.DataFrame, split: str, outcome: str
) -> pl.DataFrame:
    only_oof = (
        (
            metric_frame.filter(pl.col("metric").str.contains("out_of_fold"))
            .with_columns(pl.col("value").cast(pl.Float64))
            .group_by("run_name")
            .agg(
                pl.col("value").mean().round(3).alias("mean"),
                pl.col("value").std().round(3).alias("std"),
            )
        )
        .with_columns(
            pl.concat_str([pl.col("mean"), pl.lit(" ±"), pl.col("std")], separator="").alias(
                "pretty_value"
            )
        )
        .select("run_name", "pretty_value")
    ).with_columns(pl.lit(split).alias("split"), pl.lit(outcome).alias("outcome"))
    return only_oof


if __name__ == "__main__":
    experiment_dicts = {
        "SCZ": {"Train": "sczbp/scz_only", "Test": "sczbp/test_scz"},
        "BP": {"Train": "sczbp/bp_only", "Test": "sczbp/test_bp"},
    }
    metrics_of_interest = ["all_oof_BinaryAUROC"]

    client = MlflowClientWrapper()

    method_dfs: list[pl.DataFrame] = []
    for outcome, split_dict in experiment_dicts.items():
        for split, experiment_name in split_dict.items():
            metric_frame = client.get_all_metrics_for_experiment(
                experiment_name=experiment_name
            ).frame
            if split == "Train":
                method_df = calculate_pretty_oof_performance(
                    metric_frame=metric_frame, split=split, outcome=outcome
                )
            elif split == "Test":
                method_df = (
                    metric_frame.with_columns(
                        pl.col("value").round(2).cast(pl.String),
                        pl.lit(split).alias("split"),
                        pl.lit(outcome).alias("outcome"),
                    )
                    .drop("metric")
                    .rename({"value": "pretty_value"})
                )

            method_dfs.append(method_df)  # type: ignore

    method_df = (
        pl.concat(method_dfs, how="vertical")
        .pivot(values="pretty_value", index=["outcome", "run_name"], columns="split")
        .with_columns(
            pl.col("run_name").cast(
                pl.Enum(
                    ["structured-text-synthetic", "structured-text", "text-only", "structured-only"]
                )
            )
        )
    ).sort("outcome", "run_name")

    method_df.to_pandas().to_html(SCZ_BP_EVAL_OUTPUT_DIR / "performance_by_individual_models.html")
