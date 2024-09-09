import pandas as pd
import polars as pl
from sklearn.preprocessing import StandardScaler

from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.common.model_training.data_loader.utils import load_and_filter_split_from_cfg
from psycop.projects.forced_admission_inpatient.model_eval.selected_runs import (
    get_best_eval_pipeline,
)
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    ForcedAdmissionInpatientPipelineRun,
)


def _load_vocabulary(vocab_filename: str) -> pd.DataFrame:
    vocab_filepath = OVARTACI_SHARED_DIR / "text_models" / "vocabulary_lists" / vocab_filename

    return pd.read_parquet(vocab_filepath)


def generate_feature_coefs_table(
    pipeline_run: ForcedAdmissionInpatientPipelineRun, vocab_filename: str
) -> pl.DataFrame:
    pipeline = pipeline_run.pipeline_outputs.pipe

    # Import text model vocabulary
    vocab = _load_vocabulary(vocab_filename)

    val_df = load_and_filter_split_from_cfg(
        data_cfg=pipeline_run.inputs.cfg.data,
        pre_split_cfg=pipeline_run.inputs.cfg.preprocessing.pre_split,
        split="val",
    )

    train_df = load_and_filter_split_from_cfg(
        data_cfg=pipeline_run.inputs.cfg.data,
        pre_split_cfg=pipeline_run.inputs.cfg.preprocessing.pre_split,
        split="val",
    )

    feature_names = [c for c in train_df.columns if "pred_" in c]

    # Concatenate the train and validation DataFrames
    combined_df = pd.concat([train_df, val_df], ignore_index=True)

    # Keep only the columns that are in feature_names
    train_predictors = combined_df[feature_names]

    # Standardize the predictors (X)
    scaler = StandardScaler()
    train_predictors_scaled = scaler.fit_transform(train_predictors)

    # Get the standard deviation of each feature from the fitted StandardScaler
    feature_std_devs = scaler.scale_

    try:
        pipeline[0]["feature_selection"]  # type: ignore

        feature_indices = pipeline["preprocessing"]["feature_selection"].get_support(  # type: ignore
            indices=True
        )
        selected_feature_names = [feature_names[i] for i in feature_indices]

        selected_feature_std_devs = [feature_std_devs[i] for i in feature_indices]  # type: ignore

    except KeyError:
        selected_feature_names = feature_names  # type: ignore

    # Get standardised predictor coefficients
    standard_coefs = pipeline.named_steps["model"].coef_ * selected_feature_std_devs  # type: ignore

    # Create a DataFrame to store the feature names and their corresponding gain
    feature_table = pl.DataFrame(
        {"Feature Name": selected_feature_names, "Standardised coefficients": standard_coefs[0]}
    )

    # Sort the table by gain in descending order
    feature_table = feature_table.sort("Standardised coefficients", descending=True)

    # Get the top 15 features and the bottom 15 features and add them to a new DataFrame
    top_bottom_features = feature_table.tail(15)
    top_top_features = feature_table.head(15)

    # Concatenate the two DataFrames
    top_features = pl.concat([top_bottom_features, top_top_features]).to_pandas()

    output_path = pipeline_run.paper_outputs.paths.tables / "fa_inpatient_feature_coefficients.xlsx"

    top_features.to_excel(output_path, index=False)

    return top_features  # type: ignore


if __name__ == "__main__":
    top_100_features = generate_feature_coefs_table(
        pipeline_run=get_best_eval_pipeline(),
        vocab_filename="vocab_tfidf_psycop_train_all_sfis_preprocessed_sfi_type_all_sfis_ngram_range_12_max_df_09_min_df_2_max_features_750.parquet",
    )
