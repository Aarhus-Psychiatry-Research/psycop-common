# type: ignore
import pathlib
import re
from pathlib import Path

import pandas as pd
import polars as pl
from sklearn.pipeline import Pipeline

from psycop.common.global_utils.mlflow.mlflow_data_extraction import (
    MlflowClientWrapper,
    PsycopMlflowRun,
)

from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR

def restraint_parse_static_feature(full_string: str) -> str:
    """Takes a static feature name and returns a human readable version of it."""
    feature_name = full_string.replace("pred_", "")

    feature_capitalised = feature_name[0].upper() + feature_name[1:]

    manual_overrides = {"age_in_years": "Age", "sex_female": "Female"}

    if feature_capitalised in manual_overrides:
        feature_capitalised = manual_overrides[feature_capitalised]
    return feature_capitalised


def restraint_parse_temporal_feature(full_string: str) -> str:
    feature_name = re.findall(r"pred_(?:pred_)?(.*)?_within", full_string)[0]
    if "_" in feature_name:
        words = feature_name.split("_")
        words[0] = words[0].capitalize()
        feature_name = " ".join(word for word in words)
    else:
        feature_name = feature_name.capitalize()

    lookbehind = re.findall(r"within_(.*)?_days", full_string)[0]
    resolve_multiple = re.findall(r"days_(.*)?_fallback", full_string)[0]

    if "Tfidf" in feature_name:
        vocab = pl.read_parquet(OVARTACI_SHARED_DIR / "text_models" / "vocabulary_lists" / "vocab_tfidf_psycop_train_all_sfis_preprocessed_sfi_type_all_sfis_ngram_range_12_max_df_09_min_df_2_max_features_750.parquet")

        tfidf_idx = re.search(r"(\d+)", feature_name).group(0) # type: ignore
        tfidf_word = vocab.filter(pl.col("Index") == int(tfidf_idx))["Word"][0]
        feature_name = re.sub(tfidf_idx, tfidf_word, feature_name)

    output_string = f"{feature_name} {lookbehind}-day {resolve_multiple} "
    return output_string


def restraint_feature_name_to_readable(full_string: str) -> str:
    if "within" not in full_string:
        output_string = restraint_parse_static_feature(full_string)
    else:
        output_string = restraint_parse_temporal_feature(full_string=full_string)
    return output_string


def restraint_generate_feature_importance_table(
    pipeline: Pipeline, clf_model_name: str = "classifier"
) -> pd.DataFrame:
    # Get feature importance scores
    feature_importances = pipeline.named_steps[clf_model_name].feature_importances_

    feature_names_ = pipeline.feature_names_in_
    feature_names = feature_names_[pipeline.named_steps['feature_selection'].get_support()]

    # Create a DataFrame to store the feature names and their corresponding gain
    feature_table = pl.DataFrame(
        {"Feature Name": feature_names, "Feature Importance": feature_importances}
    )

    # Sort the table by gain in descending order
    feature_table = feature_table.sort("Feature Importance", descending=True)

    # Get the top 100 features by gain
    top_100_features = feature_table.head(100).with_columns(
        pl.col("Feature Name").apply(lambda x: restraint_feature_name_to_readable(x))
    )

    pd_df = top_100_features.to_pandas()
    pd_df = pd_df.reset_index()
    pd_df["index"] = pd_df["index"] + 1
    pd_df = pd_df.set_index("index")

    return pd_df


def restraint_feature_importance_table_facade(pipeline: Pipeline, output_dir: Path) -> None:
    feat_imp = restraint_generate_feature_importance_table(
        pipeline=pipeline, clf_model_name="classifier"
    )
    pl.Config.set_tbl_rows(100)
    (output_dir / "predictor_importance.html").write_text(feat_imp.to_html())


if __name__ == "__main__":
    run = MlflowClientWrapper().get_run("restraint_split_tuning_v2_best_run_evaluated_on_test", "magnificent-bird-866")

    feat_imp = restraint_generate_feature_importance_table(
        pipeline=run.sklearn_pipeline(), clf_model_name="classifier"
    )
    pl.Config.set_tbl_rows(100)

    pathlib.Path("restraint_feature_importances.html").write_text(feat_imp.to_html())

