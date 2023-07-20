"""Apply sentence transformer to text data and save to disk"""
from sentence_transformers import SentenceTransformer
import pandas as pd

def embed_text(model: SentenceTransformer, text: list[str]) -> pl.DataFrame:
    embeddings = model.encode(text)
    return pd.DataFrame(embeddings)



if __name__ == "__main__":
    
