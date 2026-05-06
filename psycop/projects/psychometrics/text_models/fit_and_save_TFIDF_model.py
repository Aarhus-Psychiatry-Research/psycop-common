from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_split import (
    FilterByRandom2025Splits,
)
from psycop.projects.psychometrics.text_models.text_model_pipeline import text_model_pipeline
from psycop.projects.psychometrics.text_models.text_sfi_list import (
    get_400_most_common_text_sfi_names,
)

if __name__ == "__main__":
    text_model_pipeline(
        model="tfidf",
        split_ids_presplit_step=FilterByRandom2025Splits(splits_to_keep=["train", "val"]),
        sfi_type=get_400_most_common_text_sfi_names(),
        corpus_name="psycop_psykometri_train_val_test_all_sfis_preprocessed_added_psyk_konf_2025_random_split",
        corpus_preprocessed=True,
        max_features=750,
        max_df=0.9,
        min_df=2,
        ngram_range=(1, 2),
    )
