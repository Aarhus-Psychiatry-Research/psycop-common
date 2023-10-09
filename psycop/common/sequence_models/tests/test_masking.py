from typing import Callable

import pytest
import torch
from torch import nn

from psycop.common.sequence_models import BEHRTEmbedder, BEHRTForMaskedLM

from .test_behrt import patients, trainable_module  # noqa: F401 # type: ignore
from psycop.projects.sequence_models.train import Config


@pytest.mark.parametrize(
    "embedding_module",
    [BEHRTEmbedder(d_model=32, dropout_prob=0.1, max_sequence_length=128)],
)
def test_masking_fn(patients: list, embedding_module: BEHRTEmbedder):
    """
    Test masking function
    """
    encoder_layer = nn.TransformerEncoderLayer(d_model=384, nhead=6)
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    embedding_module.fit(patients)

    config = Config()

    task = BEHRTForMaskedLM(
        embedding_module=embedding_module,
        encoder_module=encoder,
        optimizer_kwargs=config.optimization_config.optimizer_kwargs,
        lr_scheduler_kwargs=config.optimization_config.lr_scheduler_kwargs,
    )

    inputs_ids = embedding_module.collate_patients(patients)

    masked_input_ids, masked_labels = task.masking_fn(inputs_ids)

    # assert types
    assert isinstance(masked_input_ids, dict)
    assert isinstance(masked_input_ids["diagnosis"], torch.Tensor)
    assert isinstance(masked_labels, torch.Tensor)

    # Check that padding is ignored
    padding_mask = masked_input_ids["is_padding"] == 1
    assert (masked_labels[padding_mask] == -1).all()


@pytest.mark.parametrize("masking_fn", [BEHRTForMaskedLM.mask])
def test_masking_never_masks_0_elements_in_seq(masking_fn: Callable):
    # If no element in the batch is masked we get an error since the MLM module expects
    # at least one element to be masked.
    n_diagnoses_in_vocab = 4
    diagnosis = torch.randint(0, n_diagnoses_in_vocab, (2, 2))
    padding_mask = torch.zeros_like(diagnosis, dtype=torch.bool)

    input_kwargs = {
        "diagnosis": diagnosis,
        "n_diagnoses_in_vocab": n_diagnoses_in_vocab,
        "mask_token_id": 0,
        "padding_mask": padding_mask,
    }

    for _i in range(100):
        result = masking_fn(**input_kwargs)
        no_elements_are_masked = torch.all(result[1] == -1)
        assert not no_elements_are_masked
