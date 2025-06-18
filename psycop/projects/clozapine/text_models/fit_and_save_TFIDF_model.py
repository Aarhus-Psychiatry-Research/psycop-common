from psycop.projects.clozapine.text_models.text_model_pipeline import text_model_pipeline

if __name__ == "__main__":
    text_model_pipeline(
        model="tfidf",
        split_ids_presplit_step=["train", "val"],
        corpus_name="psycop_clozapine_train_val_test_all_sfis_preprocessed_added_psyk_konf",
        corpus_preprocessed=True,
        max_features=750,
        max_df=0.9,
        min_df=2,
        ngram_range=(1, 2),
    )
