# type: ignore
import pickle as pkl
import re
from pathlib import Path

import pandas as pd
import polars as pl
from sklearn.pipeline import Pipeline

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper


def scz_bp_parse_static_feature(full_string: str) -> str:
    """Takes a static feature name and returns a human readable version of it."""
    feature_name = full_string.replace("pred_", "")

    feature_capitalised = feature_name[0].upper() + feature_name[1:]

    manual_overrides = {"Age_in_years": "Age (years)"}

    if feature_capitalised in manual_overrides:
        feature_capitalised = manual_overrides[feature_capitalised]
    return feature_capitalised


def scz_bp_parse_temporal_feature(full_string: str) -> str:
    feature_name = re.findall(r"pred_(.*)?_within", full_string)[0]
    if "_disorders" in feature_name:
        words = feature_name.split("_")
        words[0] = words[0].capitalize()
        feature_name = " ".join(word for word in words)

    lookbehind = re.findall(r"within_(.*)?_days", full_string)[0]
    resolve_multiple = re.findall(r"days_(.*)?_fallback", full_string)[0]

    remove = ["all_relevant_", "aktuelt_psykisk_", r"_layer_\d_*"]
    remove = "(%s)" % "|".join(remove)

    feature_name = re.sub(remove, "", feature_name)
    output_string = f"{feature_name} {lookbehind}-day {resolve_multiple} "
    return output_string


def scz_bp_feature_name_to_readable(full_string: str) -> str:
    if "within" not in full_string:
        output_string = scz_bp_parse_static_feature(full_string)
    else:
        output_string = scz_bp_parse_temporal_feature(full_string=full_string)
    return output_string


def scz_bp_generate_feature_importance_table(
    pipeline: Pipeline, clf_model_name: str = "classifier"
) -> pd.DataFrame:
    # Get feature importance scores
    feature_importances = pipeline.named_steps[clf_model_name].feature_importances_

    if hasattr(pipeline.named_steps[clf_model_name], "feature_names"):
        selected_feature_names = pipeline.named_steps[clf_model_name].feature_names
    elif hasattr(pipeline.named_steps[clf_model_name], "feature_name_"):
        selected_feature_names = pipeline.named_steps[clf_model_name].feature_name_
    elif hasattr(pipeline.named_steps[clf_model_name], "feature_names_in_"):
        selected_feature_names = pipeline.named_steps[clf_model_name].feature_names_in_
    else:
        raise ValueError("The classifier does not implement .feature_names or .feature_name_")

    # Create a DataFrame to store the feature names and their corresponding gain
    feature_table = pl.DataFrame(
        {"Feature Name": selected_feature_names, "Feature Importance": feature_importances}
    )

    # Sort the table by gain in descending order
    feature_table = feature_table.sort("Feature Importance", descending=True)
    # Get the top 100 features by gain
    top_100_features = feature_table.head(100).with_columns(
        #  pl.col("Feature Importance").round(3),   # noqa: ERA001
        pl.col("Feature Name").apply(lambda x: scz_bp_feature_name_to_readable(x))
    )

    pd_df = top_100_features.to_pandas()
    pd_df = pd_df.reset_index()
    pd_df["index"] = pd_df["index"] + 1
    pd_df = pd_df.set_index("index")

    return pd_df


if __name__ == "__main__":
    best_experiment = "sczbp/structured_text_xgboost_ddpm_3x_positives"
    best_run = MlflowClientWrapper().get_best_run_from_experiment(
        experiment_name=best_experiment, metric="all_oof_BinaryAUROC"
    )

    with best_run.download_artifact("sklearn_pipe.pkl").open("rb") as pipe_pkl:
        pipe = pkl.load(pipe_pkl)

    feat_imp = scz_bp_generate_feature_importance_table(pipeline=pipe, clf_model_name="classifier")
    pl.Config.set_tbl_rows(100)

    with (Path(__file__).parent / f"feat_imp_100_{best_experiment.split('/')[1]}.html").open(
        "w"
    ) as html_file:
        html_file.write(feat_imp.to_html())
