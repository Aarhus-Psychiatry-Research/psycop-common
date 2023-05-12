import polars as pl
from numpy import indices
from psycop.model_training.data_loader.utils import load_and_filter_split_from_cfg
from psycop.projects.t2d.paper_outputs.config import EVAL_RUN, TABLES_PATH
from psycop.projects.t2d.utils.feature_name_to_readable import feature_name_to_readable

if __name__ == "__main__":
    pipeline = EVAL_RUN.pipe

    # Get feature importance scores
    feature_importances = pipeline.named_steps["model"].feature_importances_
    feature_indices = pipeline["preprocessing"]["feature_selection"].get_support(
        indices=True
    )

    split_df = load_and_filter_split_from_cfg(
        data_cfg=EVAL_RUN.cfg.data,
        pre_split_cfg=EVAL_RUN.cfg.preprocessing.pre_split,
        split="test",
    )

    feature_names = [c for c in split_df if "pred_" in c]
    selected_feature_names = [feature_names[i] for i in feature_indices]

    # Create a DataFrame to store the feature names and their corresponding gain
    feature_table = pl.DataFrame(
        {"Feature Name": selected_feature_names, "Gain": feature_importances}
    )

    # Sort the table by gain in descending order
    feature_table = feature_table.sort("Gain", descending=True)
    # Get the top 100 features by gain
    top_100_features = feature_table.head(100).with_columns(
        pl.col("Gain").round(3),
        pl.col("Feature Name").apply(lambda x: feature_name_to_readable(x)),
    )

    with (TABLES_PATH / "feature_importance_by_gain.html").open("w") as html_file:
        html_file.write(top_100_features.to_pandas().to_html())
