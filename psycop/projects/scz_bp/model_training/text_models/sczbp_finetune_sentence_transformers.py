"""Finetune Sentence Transformers model on different sets of notes."""

from sentence_transformers import SentenceTransformer

from psycop.common.feature_generation.loaders.raw.load_ids import (
    SplitName,
    load_stratified_by_outcome_split_ids,
    load_stratified_by_region_split_ids,
)
from psycop.common.feature_generation.text_embeddings.train_sentence_transformers import (
    train_simcse_model_from_text,
)

if __name__ == "__main__":
    DEBUG = False

    note_types_dict = {
        "aktuelt_psykisk": ["Aktuelt psykisk"],
        # "all_relevant": [
        #     "Observation af patient, Psykiatri",
        #     "Samtale med behandlingssigte",
        #     "Aktuelt psykisk",
        #     "Aktuelt socialt, Psykiatri",
        #     "Aftaler, Psykiatri",
        #     "Aktuelt somatisk, Psykiatri",
        #     "Objektivt psykisk",
        #     "Kontaktårsag",
        #     "Telefonnotat",
        #     "Semistruktureret diagnostisk interview",
        #     "Vurdering/konklusion",
        # ],
    }

    BATCH_SIZE = 12
    EPOCHS = 1
    N_ROWS = None
    TRAIN_SPLITS = [SplitName.TRAIN, SplitName.VALIDATION]
    split_id_loaders = {
        "region": load_stratified_by_region_split_ids,
        "id_outcome": load_stratified_by_outcome_split_ids,
    }
    SPLIT_TYPE = "region"

    for note_types_name, note_types in note_types_dict.items():
        model = SentenceTransformer("chcaa/dfm-encoder-large-v1")
        print(f"Training model for {note_types_name}")
        train_simcse_model_from_text(
            model=model,
            text_sfi_names=note_types,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            model_save_name=f"dfm-encoder-large-v1-{note_types_name}-finetuned-split-{SPLIT_TYPE}",
            n_rows=N_ROWS,
            train_splits=TRAIN_SPLITS,
            split_ids_loader=split_id_loaders[SPLIT_TYPE],
            debug=DEBUG,
        )
