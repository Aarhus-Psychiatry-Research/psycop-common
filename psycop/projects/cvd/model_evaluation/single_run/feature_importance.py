# type: ignore
import pickle as pkl
import re

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
    remove = "(%s)" % "|".join(remove)  # noqa

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

    feature_names = pipeline.feature_names_in_

    # Create a DataFrame to store the feature names and their corresponding gain
    feature_table = pl.DataFrame(
        {"Feature Name": feature_names, "Feature Importance": feature_importances}
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
    experiment_dict = {
        "Joint": "sczbp/test_tfidf_1000",
        "SCZ": "sczbp/test_scz",
        "BP": "sczbp/test_bp",
    }

    for outcome, experiment_name in experiment_dict.items():
        best_run = MlflowClientWrapper().get_best_run_from_experiment(
            experiment_name=experiment_name, metric="all_oof_BinaryAUROC"
        )

        with best_run.download_artifact("sklearn_pipe.pkl").open("rb") as pipe_pkl:
            pipe = pkl.load(pipe_pkl)

        feat_imp = scz_bp_generate_feature_importance_table(
            pipeline=pipe, clf_model_name="classifier"
        )
        pl.Config.set_tbl_rows(100)

        with (
            SCZ_BP_EVAL_OUTPUT_DIR / f"{outcome}_feat_imp_100_{experiment_name.split('/')[1]}.html"
        ).open("w") as html_file:
            html_file.write(feat_imp.to_html())
