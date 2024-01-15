"""Finetune Sentence Transformers model on different sets of notes."""

from typing import Literal

from sentence_transformers import SentenceTransformer

from psycop.common.feature_generation.text_embeddings.train_sentence_transformers import (
    train_simcse_model_from_text,
)
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_split import (
    FilterByOutcomeStratifiedSplits,
    RegionalFilter,
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

    BATCH_SIZE = 8
    EPOCHS = 1
    N_ROWS = 100_000
    TRAIN_SPLITS: list[Literal["train", "val", "test"]] = ["train", "val"]
    split_id_loaders = {
        "region": RegionalFilter(splits_to_keep=TRAIN_SPLITS),
        "id_outcome": FilterByOutcomeStratifiedSplits(splits_to_keep=TRAIN_SPLITS),
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
            model_save_name=f"dfm-encoder-large-v1-{note_types_name}-finetuned-split-{SPLIT_TYPE}-n_rows_{str(N_ROWS)}",
            n_rows=N_ROWS,
            split_ids_presplit_step=split_id_loaders[SPLIT_TYPE],
            debug=DEBUG,
        )
