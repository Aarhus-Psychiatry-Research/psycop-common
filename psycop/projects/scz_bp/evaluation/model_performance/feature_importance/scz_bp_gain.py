import polars as pl

from psycop.common.model_training.data_loader.utils import (
    load_and_filter_split_from_cfg,
)
from psycop.projects.scz_bp.evaluation.pipeline_objects import PipelineRun
from psycop.projects.t2d.utils.feature_name_to_readable import feature_name_to_readable


def generate_feature_importance_table(pipeline_run: PipelineRun) -> pl.DataFrame:
    pipeline = pipeline_run.pipeline_outputs.pipe

    # Get feature importance scores
    feature_importances = pipeline.named_steps["model"].feature_importances_

    split_df = load_and_filter_split_from_cfg(
        data_cfg=pipeline_run.inputs.cfg.data,
        pre_split_cfg=pipeline_run.inputs.cfg.preprocessing.pre_split,
        split="val",
    )
    feature_names = [c for c in split_df.columns if "pred_" in c]

    if hasattr(pipeline.named_steps["model"], "feature_names"):
        selected_feature_names = pipeline.named_steps["model"].feature_names
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

    pd_df = top_100_features.to_pandas()
    pd_df = pd_df.reset_index()
    pd_df["index"] = pd_df["index"] + 1
    pd_df = pd_df.set_index("index")

    with (
        pipeline_run.paper_outputs.paths.tables / "feature_importance_by_gain.html"
    ).open("w") as html_file:
        html = pd_df.to_html()
        html_file.write(html)

    return top_100_features


if __name__ == "__main__":
    from psycop.projects.scz_bp.evaluation.model_selection.performance_by_group_lookahead_model_type import (
        DEVELOPMENT_PIPELINE_RUN,
    )

    top_100_features = generate_feature_importance_table(
        pipeline_run=DEVELOPMENT_PIPELINE_RUN,
    )
