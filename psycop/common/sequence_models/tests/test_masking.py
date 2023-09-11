import pytest
import torch
from torch import nn

from psycop.common.sequence_models import BEHRTEmbedder, BEHRTForMaskedLM

from .test_main import patients, trainable_module  # noqa: F401 # type: ignore


@pytest.mark.parametrize(
    "embedding_module",
    [BEHRTEmbedder(d_model=32, dropout_prob=0.1, max_sequence_length=128)],
)
def test_masking_fn(patients: list, embedding_module: BEHRTEmbedder):  # noqa: F811
    """
    Test masking function
    """
    encoder_layer = nn.TransformerEncoderLayer(d_model=384, nhead=6)
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    embedding_module.fit(patients)

    task = BEHRTForMaskedLM(embedding_module=embedding_module, encoder_module=encoder)

    inputs_ids = embedding_module.collate_patients(patients)

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


def test_masking_never_masks_0_elements_in_seq(
    trainable_module: BEHRTForMaskedLM,
):  # noqa: F811
    # If no element in the batch is masked, mask the first element.
    # Is necessary to not get errors with small batch sizes, since the MLM module expects
    # at least one element to be masked.
    n_diagnoses_in_vocab = 4
    diagnosis = torch.randint(0, n_diagnoses_in_vocab, (2, 2))
    padded_sequence_ids = {
        "diagnosis": diagnosis,
        "is_padding": torch.zeros_like(diagnosis),
    }

    for _i in range(100):
        result = trainable_module.masking_fn(padded_sequence_ids)
        no_elements_are_masked = torch.all(result[1] == -1)
        assert not no_elements_are_masked
