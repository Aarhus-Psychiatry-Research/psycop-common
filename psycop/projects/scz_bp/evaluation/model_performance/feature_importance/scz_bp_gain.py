# type: ignore
import polars as pl
from sklearn.pipeline import Pipeline

from psycop.projects.t2d.utils.feature_name_to_readable import feature_name_to_readable


def generate_feature_importance_table(pipeline: Pipeline) -> pl.DataFrame:
    # Get feature importance scores
    feature_importances = pipeline.named_steps["model"].feature_importances_

    if hasattr(pipeline.named_steps["model"], "feature_names"):
        selected_feature_names = pipeline.named_steps["model"].feature_names

    # Create a DataFrame to store the feature names and their corresponding gain
    feature_table = pl.DataFrame(
        {"Feature Name": selected_feature_names, "Gain": feature_importances}
    )

    # Sort the table by gain in descending order
    feature_table = feature_table.sort("Gain", descending=True)
    # Get the top 100 features by gain
    top_100_features = feature_table.head(100).with_columns(
        pl.col("Gain").round(3), pl.col("Feature Name").apply(lambda x: feature_name_to_readable(x))
    )

    pd_df = top_100_features.to_pandas()
    pd_df = pd_df.reset_index()
    pd_df["index"] = pd_df["index"] + 1
    pd_df = pd_df.set_index("index")

    with (
        pipeline_run.paper_outputs.paths.tables  # noqa: F821
        / "feature_importance_by_gain.html"  # noqa: F821
    ).open("w") as html_file:
        html = pd_df.to_html()
        html_file.write(html)

    return top_100_features