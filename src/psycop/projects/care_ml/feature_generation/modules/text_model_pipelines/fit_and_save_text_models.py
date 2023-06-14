"""Pipeline for fitting and saving BoW and TF-IDF models on a preprocessed corpus"""

from psycop.common.feature_generation.text_models.text_model_pipeline import (
    text_model_pipeline,
)

if __name__ == "__main__":

    text_model_pipeline(
        model="bow",
        corpus_name="psycop_train_all_sfis_all_years_lowercase_stopwords_and_symbols_removed",
        sfi_type=["Aktuelt psykisk"],  # Current Subjective Mental State
        max_features=100,
        max_df=1.0,
        min_df=1,
        ngram_range=(1, 2),
    )

    text_model_pipeline(
        model="tfidf",
        corpus_name="psycop_train_all_sfis_all_years_lowercase_stopwords_and_symbols_removed",
        sfi_type=["Aktuelt psykisk"],  # Current Subjective Mental State
        max_features=100,
        max_df=1.0,
        min_df=1,
        ngram_range=(1, 2),
    )
