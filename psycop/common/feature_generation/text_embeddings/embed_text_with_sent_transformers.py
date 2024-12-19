"""Apply sentence transformer to text data and save to disk"""

from time import time

import polars as pl
from sentence_transformers import SentenceTransformer

from psycop.common.feature_generation.loaders.raw.load_text import load_all_notes
from psycop.common.global_utils.paths import TEXT_EMBEDDINGS_DIR


def embed_text_to_df(model: SentenceTransformer, text: list[str]) -> pl.DataFrame:
    t0 = time()
    print("Start embedding")
    embeddings = model.encode(text, batch_size=256)
    print(f"Embedding time: {time() - t0:.2f} seconds")
    return pl.DataFrame(embeddings).select(pl.all().map_alias(lambda c: c.replace("column", "emb")))


if __name__ == "__main__":
    model_str = "paraphrase-multilingual-MiniLM-L12-v2"

    all_notes = load_all_notes(n_rows=None, include_sfi_name=True)

    model = SentenceTransformer(model_str)
    embeddings = embed_text_to_df(model, all_notes["value"].to_list())

    all_notes = pl.from_pandas(all_notes).drop(columns=["value"])  # type: ignore

    embedded_notes = pl.concat([all_notes, embeddings], how="horizontal")  # type: ignore

    TEXT_EMBEDDINGS_DIR.mkdir(exist_ok=True, parents=True)

    embedded_notes.write_parquet(
        TEXT_EMBEDDINGS_DIR / f"text_embeddings_{model_str}_added_konklusion.parquet"
    )
