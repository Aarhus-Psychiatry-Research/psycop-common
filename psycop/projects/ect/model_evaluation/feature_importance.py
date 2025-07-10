# type: ignore
import pathlib
import re
from pathlib import Path

import pandas as pd
import polars as pl
from sklearn.pipeline import Pipeline

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper


def ect_parse_static_feature(full_string: str) -> str:
    """Takes a static feature name and returns a human readable version of it."""
    feature_name = full_string.replace("pred_", "")

    feature_capitalised = feature_name[0].upper() + feature_name[1:]

    manual_overrides = {"Age_days_fallback_0": "Age", "_sex_female_fallback_0": "Female"}

    if feature_capitalised in manual_overrides:
        feature_capitalised = manual_overrides[feature_capitalised]
    return feature_capitalised


def ect_parse_temporal_feature(full_string: str) -> str:
    feature_name = re.findall(r"pred_layer_[^_]+_(.*)?_within", full_string)[0]
    if "_" in feature_name:
        words = feature_name.split("_")
        words[0] = words[0].capitalize()
        feature_name = " ".join(word for word in words)

    lookbehind = re.findall(r"within_0_to_(.*)?_days", full_string)[0]
    resolve_multiple = re.findall(r"days_(.*)?_fallback", full_string)[0]

    remove = [r"_layer_[^_]+_*"]
    remove = "(%s)" % "|".join(remove)  # noqa

    feature_name = re.sub(remove, "", feature_name)
    output_string = f"{feature_name} {lookbehind}-day {resolve_multiple} "
    return output_string


def ect_feature_name_to_readable(full_string: str) -> str:
    if "within" not in full_string:
        output_string = ect_parse_static_feature(full_string)
    else:
        output_string = ect_parse_temporal_feature(full_string=full_string)
    return output_string


def ect_generate_feature_importance_table(
    pipeline: Pipeline, clf_model_name: str = "classifier"
) -> pd.DataFrame:
    # Get feature importance scores
    feature_importances = pipeline.named_steps[clf_model_name].feature_importances_

    feature_names = pipeline.feature_names_in_

    # Create a DataFrame to store the feature names and their corresponding gain
    feature_table = pl.DataFrame(
        {"Feature Name": feature_names, "Feature Importance": feature_importances}
    )

    # Sort the table by gain in descending order
    feature_table = feature_table.sort("Feature Importance", descending=True)

    # Get the top 100 features by gain
    top_100_features = feature_table.head(100).with_columns(
        pl.col("Feature Name").apply(lambda x: ect_feature_name_to_readable(x))
    )

    pd_df = top_100_features.to_pandas()
    pd_df = pd_df.reset_index()
    pd_df["index"] = pd_df["index"] + 1
    pd_df = pd_df.set_index("index")

    return pd_df


def ect_feature_importance_table_facade(pipeline: Pipeline, output_dir: Path) -> None:
    feat_imp = ect_generate_feature_importance_table(pipeline=pipeline, clf_model_name="classifier")
    pl.Config.set_tbl_rows(100)
    (output_dir / "feature_importance.html").write_text(feat_imp.to_html())


if __name__ == "__main__":
    run = MlflowClientWrapper().get_run("ECT random split test set, xgboost", "structured_text")

    feat_imp = ect_generate_feature_importance_table(
        pipeline=run.sklearn_pipeline(), clf_model_name="classifier"
    )
    pl.Config.set_tbl_rows(100)

    pathlib.Path("ect_feature_importances.html").write_text(feat_imp.to_html())
