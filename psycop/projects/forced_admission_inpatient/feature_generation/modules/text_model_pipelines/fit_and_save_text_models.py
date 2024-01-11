"""Pipeline for fitting and saving BoW and TF-IDF models on a preprocessed corpus"""

from psycop.common.feature_generation.text_models.text_model_pipeline import (
    text_model_pipeline,
)

if __name__ == "__main__":
    text_model_pipeline(
        model="tfidf",
        corpus_name="psycop_train_val_test_all_sfis_preprocessed",
        max_features=750,
        max_df=0.9,
        min_df=2,
        ngram_range=(1, 2),
    )
