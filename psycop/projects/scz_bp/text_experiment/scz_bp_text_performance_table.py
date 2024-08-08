import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.scz_bp.evaluation.configs import SCZ_BP_EVAL_OUTPUT_DIR

if __name__ == "__main__":
    df = MlflowClientWrapper().get_all_metrics_for_experiment(
        experiment_name="sczbp/text_model_selection"
    )

    only_oof = (
        (
            df.frame.filter(pl.col("metric").str.contains("out_of_fold"))
            .groupby("run_name")
            .agg(
                pl.col("value").mean().round(3).alias("mean"),
                pl.col("value").std().round(3).alias("std"),
            )
        )
        .with_columns(pl.col("run_name").str.split_exact("-", 1))
        .unnest("run_name")
        .rename({"field_0": "Notes", "field_1": "Model"})
        # .filter(~pl.col("Notes").str.contains("combined"))
        .with_columns(
            pl.concat_str([pl.col("mean"), pl.lit(" ±"), pl.col("std")], separator="").alias(
                "pretty_value"
            )
        )
    )

    table = (
        only_oof.select("Notes", "Model", "pretty_value")
        .pivot(index="Notes", columns="Model", values="pretty_value", aggregate_function=None)
        .select("Notes", "tfidf_500", "tfidf_1000", "dfm_encoder_large", "dfm_finetuned", "pse")
    )

    with (SCZ_BP_EVAL_OUTPUT_DIR / "text_model_selection_table.html").open("w") as f:
        f.write(table.to_pandas().to_html())
