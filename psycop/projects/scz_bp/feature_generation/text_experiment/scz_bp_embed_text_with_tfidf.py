import polars as pl

from psycop.common.feature_generation.text_models.encode_text_as_tfidf_scores import (
    encode_tfidf_values_to_df,
)
from psycop.common.feature_generation.text_models.text_model_paths import (
    PREPROCESSED_TEXT_DIR,
)
from psycop.common.feature_generation.text_models.utils import load_text_model
from psycop.common.global_utils.paths import TEXT_EMBEDDINGS_DIR

if __name__ == "__main__":
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
    n_features = ["500", "1000"]

    model_file_name = "tfidf_region_split_train_val_NOTE_TYPE_preprocessed_sfi_type__ngram_range_12_max_df_09_min_df_2_max_features_N_FEATURES.pkl"

    preprocessed_notes = pl.scan_parquet(
        PREPROCESSED_TEXT_DIR / "psycop_train_val_test_all_sfis_preprocessed.parquet",
    )

    for note_name_key, note_types in note_types_dict.items():
        print(f"Embedding {note_name_key}")
        preprocessed_notes = (
            pl.scan_parquet(
                PREPROCESSED_TEXT_DIR
                / "psycop_train_val_test_all_sfis_preprocessed.parquet",
            )
            .filter(pl.col("overskrift").is_in(note_types))
            .collect()
        )

        notes_metadata = preprocessed_notes.drop(columns=["value"])
        for max_features in n_features:
            model_str = model_file_name.replace("NOTE_TYPE", note_name_key).replace(
                "N_FEATURES",
                max_features,
            )
            model_name = f"tfidf-{max_features}"

            print(f"Embedding using TF-IDF model with {max_features}")
            print(f"Shape: {preprocessed_notes.shape}")
            save_path = (
                TEXT_EMBEDDINGS_DIR
                / f"text_embeddings_{note_name_key}_{model_name}.parquet"
            )
            if save_path.exists():
                print(f"Already embedded {note_name_key} with {model_str}. Skipping...")
                continue

            mdl = load_text_model(model_str)
            embeddings = encode_tfidf_values_to_df(mdl, preprocessed_notes["value"])
            embeddings = pl.concat([notes_metadata, embeddings], how="horizontal")

            embeddings.write_parquet(save_path)
