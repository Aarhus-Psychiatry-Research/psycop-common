"""Tests for generating huggingface embeddings."""

_test_hf_embeddings = list(range(384))


TEST_HF_EMBEDDINGS = [
    "embedding-" + str(dimension) for dimension in _test_hf_embeddings
]
