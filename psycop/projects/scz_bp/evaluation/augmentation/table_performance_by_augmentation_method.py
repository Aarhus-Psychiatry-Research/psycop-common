from pathlib import Path


import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.scz_bp.evaluation.configs import SCZ_BP_EVAL_OUTPUT_DIR

if __name__ == "__main__":
    save_dir = Path(__file__).parent

    experiment_dicts = {
        "TabDDPM": "sczbp/structured_text_xgboost_ddpm",
        "SMOTE": "sczbp/structured_text_xgboost_smote",
    }
    metrics_of_interest = ["sample_multiplier", "all_oof_BinaryAUROC"]

    client = MlflowClientWrapper()

    method_dfs: list[pl.DataFrame] = []
    for method, experiment in experiment_dicts.items():
        metric_frame = client.get_all_metrics_for_experiment(experiment_name=experiment)

        only_oof = (
            metric_frame.frame.filter(
                pl.col(metric_frame.metric_col_name).str.contains("out_of_fold")
            )
            .group_by(metric_frame.run_name_col_name)
            .agg(
                pl.col("value").mean().round(3).alias("mean"),
                pl.col("value").std().round(3).alias("std"),
            )
        ).with_columns(
            pl.concat_str([pl.col("mean"), pl.lit(" ±"), pl.col("std")], separator="").alias(
                "pretty_value"
            )
        ).select("run_name", "pretty_value")

        df = (
            (
                metric_frame.frame.filter(
                    pl.col(metric_frame.metric_col_name) == "sample_multiplier")
                ).join(only_oof, on="run_name").select(pl.col("value").alias("sample_multiplier"), "pretty_value")
                .with_columns(pl.lit(method).alias("Method"),
                    pl.col("sample_multiplier").cast(pl.Int32)).sort("sample_multiplier")
                .pivot(
                    values="pretty_value",
                    index="Method",
                    columns="sample_multiplier",
                )

            )
        method_dfs.append(df)

    method_df = pl.concat(method_dfs, how="vertical")
    method_df.to_pandas().to_html(SCZ_BP_EVAL_OUTPUT_DIR / "performance_by_augmentation_method_table.html")
