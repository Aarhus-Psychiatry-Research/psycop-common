"""Apply sentence transformer to text data and save to disk"""

from time import time

import polars as pl
from sentence_transformers import SentenceTransformer

from psycop.projects.forced_admission_inpatient_temp_val.feature_generation.modules.loaders.load_text_fa_2025 import (
    get_valid_text_sfi_names,
    load_text,
)
from psycop.projects.forced_admission_inpatient_temp_val.feature_generation.modules.text_models.forced_adm_temp_val_text_model_paths import (
    TEXT_EMBEDDINGS_DIR,
)


def embed_text_to_df(model: SentenceTransformer, text: list[str]) -> pl.DataFrame:
    t0 = time()
    print("Start embedding")
    embeddings = model.encode(text, batch_size=256)
    print(f"Embedding time: {time() - t0:.2f} seconds")
    return pl.DataFrame(embeddings).select(pl.all().map_alias(lambda c: c.replace("column", "emb")))


if __name__ == "__main__":
    model_str = "paraphrase-multilingual-MiniLM-L12-v2"

    text_sfis = get_valid_text_sfi_names()

    text = load_text(text_sfi_names=text_sfis, include_sfi_name=True)

    model = SentenceTransformer(model_name_or_path=model_str)
    embeddings = embed_text_to_df(model, text["value"].to_list())

    text = pl.from_pandas(text).drop(columns=["value"])  # type: ignore

    embedded_notes = pl.concat([text, embeddings], how="horizontal")  # type: ignore

    TEXT_EMBEDDINGS_DIR.mkdir(exist_ok=True, parents=True)

    embedded_notes.write_parquet(
        TEXT_EMBEDDINGS_DIR
        / f"fa_temp_val_text_transformer_embeddings_{model_str}_2020_2025.parquet"
    )
