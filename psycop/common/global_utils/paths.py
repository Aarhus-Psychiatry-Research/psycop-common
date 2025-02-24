from pathlib import Path

PSYCOP_PKG_ROOT = Path(__file__).parent.parent
OVARTACI_SHARED_DIR = Path(r"E:\shared_resources")
TEXT_EMBEDDING_MODELS_DIR = OVARTACI_SHARED_DIR / "text_embedding_models"
TEXT_EMBEDDINGS_DIR = (
    OVARTACI_SHARED_DIR / "text_embeddings"
)  # empty now but preserved for maintaining integrity of old projects - see PR #1051
RANDOM_SEED = 42
