# type: ignore
import re
from pathlib import Path

import pandas as pd
import polars as pl
from sklearn.pipeline import Pipeline

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.projects.clozapine.model_eval.config import CLOZAPINE_EVAL_OUTPUT_DIR


def load_vocab(path: Path) -> pl.DataFrame:
    """Load a .pkl vocabulary file (Word -> Index) into a Polars DataFrame."""
    vocab_pd = pd.read_pickle(path)  # Load pickle via pandas
    vocab = pl.from_pandas(vocab_pd)  # Convert to Polars

    # Ensure correct dtypes
    vocab = vocab.with_columns([pl.col("Word").cast(pl.Utf8), pl.col("Index").cast(pl.Int64)])

    return vocab


def clozapine_parse_static_feature(full_string: str) -> str:
    """Takes a static feature name and returns a human readable version of it."""
    feature_name = full_string.replace("pred_", "")

    feature_capitalised = feature_name[0].upper() + feature_name[1:]

    manual_overrides = {"age_in_years": "Age", "sex_female": "Female"}

    if feature_capitalised in manual_overrides:
        feature_capitalised = manual_overrides[feature_capitalised]
    return feature_capitalised


def clozapine_parse_temporal_feature(full_string: str) -> str:
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
        vocab_path = (
            OVARTACI_SHARED_DIR
            / "clozapine"
            / "text_models"
            / "vocabulary_lists"
            / "vocab_tfidf_psycop_clozapine_preprocessed_added_psyk_konf_added_2025_random_split_train_val_sfi_type_all_sfis_ngram_range_12_max_df_09_min_df_2_max_features_750.pkl"
        )

        vocab = load_vocab(vocab_path)

        # Extract index from feature name, e.g. "Tfidf_1234_mean" → "1234"
        tfidf_idx = int(re.search(r"(\d+)", feature_name).group(0))

        # Lookup word from vocabulary
        tfidf_word = vocab.filter(pl.col("Index") == tfidf_idx)["Word"][0]

        # Replace index with the actual word
        feature_name = re.sub(str(tfidf_idx), tfidf_word, feature_name)

    output_string = f"{feature_name} {lookbehind}-day {resolve_multiple} "
    return output_string


def clozapine_feature_name_to_readable(full_string: str) -> str:
    if "within" not in full_string:
        output_string = clozapine_parse_static_feature(full_string)
    else:
        output_string = clozapine_parse_temporal_feature(full_string=full_string)
    return output_string


def clozapine_generate_feature_importance_table(
    pipeline: Pipeline, clf_model_name: str = "classifier"
) -> pd.DataFrame:
    # Get feature importance scores
    feature_importances = pipeline.named_steps[clf_model_name].feature_importances_

    feature_names_ = pipeline.feature_names_in_

    if "feature_selection" in pipeline.named_steps:
        support_mask = pipeline.named_steps["feature_selection"].get_support()
        feature_names = feature_names_[support_mask]
    else:
        # No feature selection in pipeline → use all input features
        feature_names = feature_names_

    # Create a DataFrame to store the feature names and their corresponding gain
    feature_table = pl.DataFrame(
        {"Feature Name": feature_names, "Feature Importance": feature_importances}
    )

    # Sort the table by gain in descending order
    feature_table = feature_table.sort("Feature Importance", descending=True)

    # Get the top 100 features by gain
    top_100_features = feature_table.head(100).with_columns(
        pl.col("Feature Name").apply(lambda x: clozapine_feature_name_to_readable(x))
    )

    pd_df = top_100_features.to_pandas()
    pd_df = pd_df.reset_index()
    pd_df["index"] = pd_df["index"] + 1
    pd_df = pd_df.set_index("index")

    return pd_df


def clozapine_feature_importance_table_facade(pipeline: Pipeline, output_dir: Path) -> None:
    feat_imp = clozapine_generate_feature_importance_table(
        pipeline=pipeline, clf_model_name="classifier"
    )
    output_dir = Path(CLOZAPINE_EVAL_OUTPUT_DIR / model)
    output_dir.mkdir(parents=True, exist_ok=True)
    pl.Config.set_tbl_rows(100)
    feat_imp.to_excel(output_dir / "predictor_importance.xlsx")

    feat_imp.to_csv(output_dir / "predictor_importance.csv")


if __name__ == "__main__":
    model = "clozapine hparam, only_structured_365d_lookahead, xgboost, 1 year lookbehind filter, 2025_random_split"
    run = MlflowClientWrapper().get_run(model, "fearless-whale-268")

    clozapine_feature_importance_table_facade(
        pipeline=run.sklearn_pipeline(), output_dir=CLOZAPINE_EVAL_OUTPUT_DIR
    )
