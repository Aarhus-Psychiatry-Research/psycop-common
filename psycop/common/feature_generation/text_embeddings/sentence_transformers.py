"""Apply sentence transformer to text data and save to disk"""
import pandas as pd
from sentence_transformers import SentenceTransformer

from psycop.common.feature_generation.loaders.raw.load_text import load_all_notes
from psycop.common.global_utils.paths import TEXT_EMBEDDINGS_DIR


def embed_text_to_df(model, text: list[str]) -> pd.DataFrame:
    embeddings = model.encode(text)
    return pd.DataFrame(embeddings)



if __name__ == "__main__":
    model_str = "paraphrase-multilingual-MiniLM-L12-v2"

    all_notes = load_all_notes(n_rows=100, include_sfi_name=True)

    model = SentenceTransformer(model_str)    
    embeddings = embed_text_to_df(model, all_notes["value"].tolist())
    
    all_notes = all_notes.drop(columns=["value"])
    all_notes = pd.concat([all_notes, embeddings], axis=1)

    all_notes.to_parquet(TEXT_EMBEDDINGS_DIR / f"text_embeddings_{model_str}.parquet")