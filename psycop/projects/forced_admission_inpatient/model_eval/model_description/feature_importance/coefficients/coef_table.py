import pandas as pd
import polars as pl
from sklearn.preprocessing import StandardScaler

from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.common.model_training.data_loader.utils import (
    load_and_filter_split_from_cfg,
)
from psycop.projects.forced_admission_inpatient.model_eval.selected_runs import (
    get_best_eval_pipeline,
)
from psycop.projects.forced_admission_inpatient.utils.feature_name_to_readable import (
    feature_name_to_readable,
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

        selected_feature_std_devs = [feature_std_devs[i] for i in feature_indices] # type: ignore

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

    # Get the top 100 features by gain
    top_100_features = feature_table.head(100).with_columns(
        pl.col("Gain").round(3),
        pl.col("Feature Name").apply(lambda x: feature_name_to_readable(x)),  # type: ignore
    )

    pd_df = top_100_features.to_pandas()
    pd_df = pd_df.reset_index()
    pd_df["index"] = pd_df["index"] + 1
    pd_df = pd_df.set_index("index")

    # Map tfidf indices with actual ngrams from vocabulary
    pd_df["Feature Name"][pd_df["Feature Name"].str.contains("tfidf")] = pd_df["Feature Name"][
        pd_df["Feature Name"].str.contains("tfidf")
    ].str.replace(r"\d+$", lambda x: vocab.loc[int(x.group())]["Word"])

    with (pipeline_run.paper_outputs.paths.tables / "feature_importance_by_gain.html").open(
        "w"
    ) as html_file:
        html = pd_df.to_html()
        html_file.write(html)

    return top_100_features

if __name__ == "__main__":
    top_100_features = generate_feature_coefs_table(
        pipeline_run=get_best_eval_pipeline(),
        vocab_filename="vocab_tfidf_psycop_train_all_sfis_preprocessed_sfi_type_all_sfis_ngram_range_12_max_df_09_min_df_2_max_features_750.parquet",
    )
