import polars as pl
from psycop.common.model_training.data_loader.utils import (
    load_and_filter_split_from_cfg,
)
from psycop.projects.t2d.paper_outputs.selected_runs import BEST_EVAL_PIPELINE
from psycop.projects.t2d.utils.feature_name_to_readable import feature_name_to_readable
from psycop.projects.t2d.utils.pipeline_objects import PipelineRun


def generate_feature_importance_table(pipeline_run: PipelineRun) -> pl.DataFrame:
    pipeline = pipeline_run.pipeline_outputs.pipe

    # Get feature importance scores
    feature_importances = pipeline.named_steps["model"].feature_importances_

    split_df = load_and_filter_split_from_cfg(
        data_cfg=pipeline_run.inputs.cfg.data,
        pre_split_cfg=pipeline_run.inputs.cfg.preprocessing.pre_split,
        split="test",
    )
    feature_names = [c for c in split_df.columns if "pred_" in c]

    if "feature_selection" in pipeline["preprocessing"]:  # type: ignore
        feature_indices = pipeline["preprocessing"]["feature_selection"].get_support(  # type: ignore
            indices=True,
        )
        selected_feature_names = [feature_names[i] for i in feature_indices]
    else:
        selected_feature_names = feature_names

    # Create a DataFrame to store the feature names and their corresponding gain
    feature_table = pl.DataFrame(
        {"Feature Name": selected_feature_names, "Gain": feature_importances},
    )

    # Sort the table by gain in descending order
    feature_table = feature_table.sort("Gain", descending=True)
    # Get the top 100 features by gain
    top_100_features = feature_table.head(100).with_columns(
        pl.col("Gain").round(3),
        pl.col("Feature Name").apply(lambda x: feature_name_to_readable(x)),  # type: ignore
    )

    with (
        pipeline_run.paper_outputs.paths.tables / "feature_importance_by_gain.html"
    ).open("w") as html_file:
        html_file.write(top_100_features.to_pandas().to_html())

    return top_100_features


if __name__ == "__main__":
    top_100_features = generate_feature_importance_table(
        pipeline_run=BEST_EVAL_PIPELINE,
    )
    pass
