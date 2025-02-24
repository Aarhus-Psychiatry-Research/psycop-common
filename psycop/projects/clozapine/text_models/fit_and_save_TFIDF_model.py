from psycop.projects.clozapine.text_models.text_model_pipeline import text_model_pipeline

if __name__ == "__main__":
    text_model_pipeline(
        model="tfidf",
        corpus_name="psycop_clozapine_train_val_all_sfis_preprocessed_v3",
        corpus_preprocessed=True,
        max_features=750,
        max_df=0.9,
        min_df=2,
        ngram_range=(1, 2),
    )
