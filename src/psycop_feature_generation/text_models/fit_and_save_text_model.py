"""Fit and save a model"""
from psycop_feature_generation.text_models.text_models_pipeline import (
    bow_model_pipeline,
    tfidf_model_pipeline,
    lda_model_pipeline,
)

if __name__ == "__main__":
    tfidf_model_pipeline(
        view="psycop_train_val_all_sfis_all_years_lowercase_stopwords_and_symbols_removed",
        # sfi_type=["Aktuelt psykisk"],
    )
