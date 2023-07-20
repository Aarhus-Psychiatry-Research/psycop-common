"""Train sentence transformer model using SimCSE loss on our data."""
from typing import Literal, Sequence

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from psycop.common.feature_generation.loaders.raw.load_text import load_text_split
from psycop.common.global_utils.paths import TEXT_EMBEDDING_MODELS_DIR


def get_train_text(
    n_rows: int | None,
    text_sfi_names: str | list[str],
    train_splits: Sequence[Literal["train", "val"]],
) -> list[str]:
    text_df = load_text_split(
        text_sfi_names=text_sfi_names, n_rows=n_rows, split_name=train_splits
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


def make_data_loader(train_data: list[InputExample], batch_size: int) -> DataLoader:
    return DataLoader(train_data, batch_size=batch_size, shuffle=True)  # type: ignore


def train_simcse_model(
    dataloader: DataLoader, model: SentenceTransformer, epochs: int, model_name: str
):
    save_dir = TEXT_EMBEDDING_MODELS_DIR / model_name
    save_dir.mkdir(exist_ok=True, parents=True)
    # use SimCSE loss for unsupervised training
    train_loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=epochs,
        show_progress_bar=True,
    )
    model.save(str(save_dir))


def train_simcse_model_from_text(
    model: SentenceTransformer,
    text_sfi_names: str | list[str],
    epochs: int,
    batch_size: int,
    model_save_name: str,
    n_rows: int | None = None,
    train_splits: Sequence[Literal["train", "val"]] = ["train"],
    debug: bool = False,
) -> None:
    if debug:
        train_text = _get_debug_text()
    else:
        train_text = get_train_text(
            n_rows=n_rows,
            text_sfi_names=text_sfi_names,
            train_splits=train_splits,
        )
    train_data = convert_list_of_texts_to_sentence_pairs(texts=train_text)
    dataloader = make_data_loader(train_data=train_data, batch_size=batch_size)
    train_simcse_model(
        dataloader=dataloader, model=model, epochs=epochs, model_name=model_save_name
    )


if __name__ == "__main__":
    DEBUG = True
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
    EPOCHS = 1
    N_ROWS = 200
    TRAIN_SPLITS: Sequence[Literal["train", "val"]] = ["train"]
    # Exp 1: continue pretraining paraphrase-multilingual-MiniLM-L12-v2
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    # Exp 2: continue pretraining model larger sequence length (e.g.
    # NbAiLab/nb-roberta-base-scandi or chcaa/dfm-encoder-large-v1)
    # model = SentenceTransformer("NbAiLab/nb-roberta-base-scandi")

    train_simcse_model_from_text(
        model=model,
        text_sfi_names=TRAIN_SFIS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        model_save_name=f"{model}-finetuned-debug-{str(DEBUG)}",
        n_rows=N_ROWS,
        train_splits=TRAIN_SPLITS,
        debug=DEBUG,
    )
