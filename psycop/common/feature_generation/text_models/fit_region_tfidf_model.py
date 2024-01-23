import polars as pl

from psycop.common.feature_generation.text_models.fit_text_models import fit_text_model
from psycop.common.feature_generation.text_models.text_model_paths import (
    PREPROCESSED_TEXT_DIR,
    TEXT_MODEL_DIR,
)
from psycop.common.feature_generation.text_models.text_model_pipeline import (
    create_model_filename,
)
from psycop.common.feature_generation.text_models.utils import (
    save_text_model_to_shared_dir,
)
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_split import (
    RegionalFilter,
)

if __name__ == "__main__":
    ## train 2 TF-IDF models; 500 and 1000 features - on both aktuelt psykisk and all SFIs

    n_features = [500, 1000]
    note_types_dict = {
        "aktuelt_psykisk": ["Aktuelt psykisk"],
        "all_relevant": [
            "Observation af patient, Psykiatri",
            "Samtale med behandlingssigte",
            "Aktuelt psykisk",
            "Aktuelt socialt, Psykiatri",
            "Aftaler, Psykiatri",
            "Aktuelt somatisk, Psykiatri",
            "Objektivt psykisk",
            "Kontaktårsag",
            "Telefonnotat",
            "Semistruktureret diagnostisk interview",
            "Vurdering/konklusion",
        ],
    }

    all_preprocessed_text = pl.scan_parquet(
        PREPROCESSED_TEXT_DIR / "psycop_train_val_test_all_sfis_preprocessed.parquet",
    )
    region_split_filter = RegionalFilter(splits_to_keep=["train", "val"])
    all_preprocessed_text = region_split_filter.apply(all_preprocessed_text)

    min_df = 2
    max_df = 0.9
    ngram_range = (1, 2)

    for note_name_key, note_types in note_types_dict.items():
        print(f"Fitting TF-IDF models on {note_name_key}")
        # filter columns
        sub_df = all_preprocessed_text.filter(
            pl.col("overskrift").is_in(note_types)
        ).collect()
        print(sub_df.shape)
        for max_features in n_features:
            print(f"Fitting model with {max_features} features")
            model_filename = create_model_filename(
                model="tfidf",
                corpus_name=f"region_split_train_val_{note_name_key}_preprocessed",
                sfi_type=[""],
                max_df=max_df,
                min_df=min_df,
                max_features=max_features,
                ngram_range=ngram_range,
            )
            model_path = TEXT_MODEL_DIR / model_filename
            if model_path.exists():
                print("Model already exists. Skipping..")

            mdl = fit_text_model(
                model="tfidf",
                corpus=sub_df["value"],
                ngram_range=ngram_range,
                max_df=max_df,
                min_df=min_df,
                max_features=max_features,
            )
            save_text_model_to_shared_dir(model=mdl, filename=model_filename)
