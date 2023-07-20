"""Apply sentence transformer to text data and save to disk"""
import polars as pl
from sentence_transformers import SentenceTransformer

from psycop.common.feature_generation.loaders.raw.load_text import load_all_notes
from psycop.common.global_utils.paths import TEXT_EMBEDDINGS_DIR


def embed_text_to_df(model: SentenceTransformer, text: list[str]) -> pl.DataFrame:
    embeddings = model.encode(text)
    return pl.DataFrame(embeddings).select(pl.all().map_alias(lambda c: c.replace("column", "emb")))



if __name__ == "__main__":
    model_str = "paraphrase-multilingual-MiniLM-L12-v2"

    all_notes = pl.from_pandas(load_all_notes(n_rows=100, include_sfi_name=True))

    model = SentenceTransformer(model_str)    
    embeddings = embed_text_to_df(model, all_notes["value"].to_list())
    
    all_notes = all_notes.drop(columns=["value"])

    embedded_notes = pl.concat([all_notes, embeddings], how="horizontal")

    embedded_notes.write_parquet(TEXT_EMBEDDINGS_DIR / f"text_embeddings_{model_str}.parquet")