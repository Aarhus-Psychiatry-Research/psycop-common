import pytest
import torch

from psycop.common.sequence_models import BEHRTEmbedder, Embedder

from .test_main import patients  # noqa: F401 # type: ignore


@pytest.mark.parametrize(
    "embedding_module",
    [BEHRTEmbedder(d_model=384, dropout_prob=0.1, max_sequence_length=128)],
)
def test_embeddings(patients: list, embedding_module: Embedder):
    """
    Test embedding interface
    """
    embedding_module.fit(patients)

    inputs_ids = embedding_module.collate_fn(patients)

    assert isinstance(inputs_ids, dict)
    assert isinstance(inputs_ids["diagnosis"], torch.Tensor)  # type: ignore
    assert isinstance(inputs_ids["age"], torch.Tensor)  # type: ignore
    assert isinstance(inputs_ids["segment"], torch.Tensor)
    assert isinstance(inputs_ids["position"], torch.Tensor)

    # forward
    outputs = embedding_module(inputs_ids)
