"""Apply sentence transformer to text data and save to disk"""
from time import time

import polars as pl
from sentence_transformers import SentenceTransformer

from psycop.common.feature_generation.loaders.raw.load_text import (
    load_text_sfis,
)
from psycop.common.global_utils.paths import (
    TEXT_EMBEDDING_MODELS_DIR,
    TEXT_EMBEDDINGS_DIR,
)


def embed_text_to_df(
    model: SentenceTransformer,
    text: list[str],
    batch_size: int = 256,
) -> pl.DataFrame:
    t0 = time()
    print("Start embedding")
    embeddings = model.encode(text, batch_size=batch_size)
    print(f"Embedding time: {time() - t0:.2f} seconds")
    return pl.DataFrame(embeddings).select(
        pl.all().map_alias(lambda c: c.replace("column", "emb")),
    )


mock_df = pl.DataFrame(
    {
        "dw_ek_borger": ["1", "2", "3"],
        "value": ["hej med dig", "min fine ven", "goddag"],
        "timestamp": ["2021-01-01", "2021-01-02", "2021-01-03"],
    },
)


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

    models = {
        "dfm-encoder-large": "chcaa/dfm-encoder-large-v1",
        "e5-large": "intfloat/multilingual-e5-large",
    }

    for note_name_key, note_types in note_types_dict.items():
        print(f"Embedding {note_name_key}")
        notes = pl.from_pandas(
            load_text_sfis(text_sfi_names=note_types, include_sfi_name=False),
        )
        notes_metadata = notes.drop(columns=["value"])

        for model_name, model_str in models.items():
            print(f"Embedding using {model_name}")
            model = SentenceTransformer(model_str)
            embeddings = embed_text_to_df(model, notes["value"].to_list())
            embeddings = pl.concat([notes_metadata, embeddings], how="horizontal")

            TEXT_EMBEDDINGS_DIR.mkdir(exist_ok=True, parents=True)
            embeddings.write_parquet(
                TEXT_EMBEDDINGS_DIR
                / f"text_embeddings_{note_name_key}_{model_name}.parquet",
            )
        # embed using local finetuned models
        print("Embedding using finetuned model")
        finetuned_model_name = (
            TEXT_EMBEDDING_MODELS_DIR
            / f"dfm-encoder-large-v1-{note_name_key}-finetuned-split-region-n_rows_100000"
        )
        model = SentenceTransformer(str(finetuned_model_name))
        embeddings = embed_text_to_df(model, notes["value"].to_list())
        embeddings = pl.concat([notes_metadata, embeddings], how="horizontal")
        embeddings.write_parquet(
            TEXT_EMBEDDINGS_DIR
            / f"text_embeddings_{note_name_key}_dfm-encoder-large-v1-finetuned.parquet",
        )
