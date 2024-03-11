# type: ignore
import polars as pl
from sklearn.pipeline import Pipeline

from psycop.projects.t2d.utils.feature_name_to_readable import feature_name_to_readable
from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
import pickle as pkl


def generate_feature_importance_table(pipeline: Pipeline) -> pl.DataFrame:
    # Get feature importance scores
    feature_importances = pipeline.named_steps["model"].feature_importances_

    if hasattr(pipeline.named_steps["model"], "feature_names"):
        selected_feature_names = pipeline.named_steps["model"].feature_names

    # Create a DataFrame to store the feature names and their corresponding gain
    feature_table = pl.DataFrame(
        {"Feature Name": selected_feature_names, "Feature Importance": feature_importances}
    )

    # Sort the table by gain in descending order
    feature_table = feature_table.sort("Feature Importance", descending=True)
    # Get the top 100 features by gain
    top_100_features = feature_table.head(100).with_columns(
        pl.col("Feature Importance").round(3),
        pl.col("Feature Name").apply(lambda x: feature_name_to_readable(x)),
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


if __name__ == "__main__":
    best_experiment = "sczbp/text_only"

    best_run = MlflowClientWrapper().get_best_run_from_experiment(
        experiment_name=best_experiment, metric="all_oof_BinaryAUROC"
    )

    with open(best_run.download_artifact("sklearn_pipe.pkl"), "rb") as pipe_pkl:
        pipe = pkl.load(pipe_pkl)
 
