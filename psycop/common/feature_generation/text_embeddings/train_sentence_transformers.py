"""Train sentence transformer model using SimCSE loss on our data."""
from collections.abc import Sequence
from time import time
from typing import Callable, Literal

import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from psycop.common.feature_generation.loaders.raw.load_ids import SplitName, load_stratified_by_outcome_split_ids, load_stratified_by_region_split_ids
from psycop.common.feature_generation.loaders.raw.load_text import load_text_split
from psycop.common.global_utils.paths import TEXT_EMBEDDING_MODELS_DIR


def get_train_text(
    n_rows: int | None,
    text_sfi_names: str | list[str],
    train_splits: Sequence[SplitName],
    split_ids_loader: Callable[[SplitName], pd.DataFrame] | None,
) -> list[str]:
    text_df = load_text_split(
        text_sfi_names=text_sfi_names,
        n_rows=n_rows,
        split_name=train_splits,
        split_ids_loader=split_ids_loader,
    )
    return text_df["value"].tolist()


def _get_debug_text() -> list[str]:
    return [
        "Your set of sentences",
        "Model will automatically add the noise",
        "And re-construct it",
        "You should provide at least 1k sentences",
    ]


def convert_list_of_texts_to_sentence_pairs(texts: list[str]) -> list[InputExample]:
    return [InputExample(texts=[s, s]) for s in texts]


def make_data_loader(train_data: list[InputExample], batch_size: int) -> DataLoader:  # type: ignore
    return DataLoader(train_data, batch_size=batch_size, shuffle=True)  # type: ignore


def train_simcse_model(
    dataloader: DataLoader,  # type: ignore
    model: SentenceTransformer,
    epochs: int,
    model_name: str,
):
    save_dir = TEXT_EMBEDDING_MODELS_DIR / model_name
    save_dir.mkdir(exist_ok=True, parents=True)
    # use SimCSE loss for unsupervised training
    train_loss = losses.MultipleNegativesRankingLoss(model)
    t0 = time()
    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=epochs,
        show_progress_bar=True,
    )
    model.save(str(save_dir))
    print(f"Model saved to {save_dir}")
    print(f"Time taken: {time() - t0}")


def train_simcse_model_from_text(
    model: SentenceTransformer,
    text_sfi_names: str | list[str],
    epochs: int,
    batch_size: int,
    model_save_name: str,
    n_rows: int | None = None,
    train_splits: Sequence[SplitName] = [SplitName.TRAIN],
    split_ids_loader: Callable[[SplitName], pd.DataFrame] | None = None,
    debug: bool = False,
) -> None:
    if debug:
        train_text = _get_debug_text()
    else:
        train_text = get_train_text(
            n_rows=n_rows,
            text_sfi_names=text_sfi_names,
            train_splits=train_splits,
            split_ids_loader=split_ids_loader,
        )
    train_data = convert_list_of_texts_to_sentence_pairs(texts=train_text)
    dataloader = make_data_loader(train_data=train_data, batch_size=batch_size)
    train_simcse_model(
        dataloader=dataloader,
        model=model,
        epochs=epochs,
        model_name=model_save_name,
    )


if __name__ == "__main__":
    DEBUG = False
    TRAIN_SFIS = [
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
    ]
    BATCH_SIZE = 128
    EPOCHS = 2
    N_ROWS = None
    TRAIN_SPLITS = [SplitName.TRAIN]
    split_id_loaders = {
        "region": load_stratified_by_region_split_ids,
        "id_outcome": load_stratified_by_outcome_split_ids,
    }
    SPLIT_TYPE = "region"
    MODEL = "miniLM"
    # Exp 1: continue pretraining paraphrase-multilingual-MiniLM-L12-v2
    model_options = {
        "miniLM": "paraphrase-multilingual-MiniLM-L12-v2",
        "scandi": "NbAiLab/nb-roberta-base-scandi",
    }
    model = SentenceTransformer(model_options[MODEL])

    train_simcse_model_from_text(
        model=model,
        text_sfi_names=TRAIN_SFIS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        model_save_name=f"{model_options[MODEL]}-finetuned-debug-{DEBUG!s}",
        n_rows=N_ROWS,
        train_splits=TRAIN_SPLITS,
        split_ids_loader=split_id_loaders[SPLIT_TYPE],
        debug=DEBUG,
    )
