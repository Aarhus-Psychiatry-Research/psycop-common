"""Finetune Sentence Transformers model on different sets of notes."""

from sentence_transformers import SentenceTransformer

from psycop.common.feature_generation.loaders.raw.load_ids import SplitName
from psycop.common.feature_generation.text_embeddings.train_sentence_transformers import (
    train_simcse_model_from_text,
)

if __name__ == "__main__":
    DEBUG = False

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

    BATCH_SIZE = 128
    EPOCHS = 1
    N_ROWS = None
    TRAIN_SPLITS = [SplitName.TRAIN, SplitName.VALIDATION]
    SPLIT_TYPE = "geographical"
    model = SentenceTransformer("chcaa/dfm-encoder-large-v1")

    for note_types_name, note_types in note_types_dict.items():
        print(f"Training model for {note_types_name}")
        train_simcse_model_from_text(
            model=model,
            text_sfi_names=note_types,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            model_save_name=f"dfm-encoder-large-v1-{note_types_name}-finetuned",
            n_rows=N_ROWS,
            train_splits=TRAIN_SPLITS,
            split_type=SPLIT_TYPE,
            debug=DEBUG,
        )
