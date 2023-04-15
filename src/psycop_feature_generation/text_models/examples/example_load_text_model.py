from psycop_feature_generation.text_models.load_text_models import _load_bow_model

if __name__ == "main": 
    bow = _load_bow_model(filename="bow_psycop_train_val_all_sfis_all_years_lowercase_stopwords_and_symbols_removed_ngram_range_11_max_df_095_min_df_2_max_features_500.pkl")