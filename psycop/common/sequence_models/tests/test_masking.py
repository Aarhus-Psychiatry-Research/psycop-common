import pytest
import torch
from torch import nn

from psycop.common.sequence_models import BEHRTEmbedder, BEHRTForMaskedLM, Embedder

from .test_main import patients  # noqa: F401 # type: ignore


@pytest.mark.parametrize(
    "embedding_module",
    [BEHRTEmbedder(d_model=32, dropout_prob=0.1, max_sequence_length=128)],
)
def test_masking_fn(patients: list, embedding_module: Embedder):
    """
    Test masking function
    """
    emb = BEHRTEmbedder(d_model=384, dropout_prob=0.1, max_sequence_length=128)
    encoder_layer = nn.TransformerEncoderLayer(d_model=384, nhead=6)
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    emb.fit(patients)

    task = BEHRTForMaskedLM(embedding_module=emb, encoder_module=encoder)

    inputs_ids = emb.collate_patients(patients)

    masked_input_ids, masked_labels = task.masking_fn(inputs_ids)

    # assert types
    assert isinstance(masked_input_ids, dict)
    assert isinstance(masked_input_ids["diagnosis"], torch.Tensor)
    assert isinstance(masked_labels, torch.Tensor)

    # assert that the masked labels are same as the input ids where they are not masked? # TODO

    # Check that padding is ignored?
    padding_mask = masked_input_ids["is_padding"] == 1
    # check that all masked_labels where padding_mask is True are -1
    assert (masked_labels[padding_mask] == -1).all()
