from datetime import datetime
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from psycop.common.data_structures import Patient, TemporalEvent
from psycop.common.sequence_models import (
    BEHRTEmbedder,
    BEHRTMaskingTask,
    Embedder,
    PatientDataset,
    Trainer,
)


@pytest.fixture
def patients() -> list[Patient]:
    """
    Returns a list of patient objects
    """

    e1 = TemporalEvent(
        timestamp=datetime(2021, 1, 1),
        value="d1",
        source_type="diagnosis",
        source_subtype="",
    )
    e2 = TemporalEvent(
        timestamp=datetime(2021, 1, 3),
        value="d2",
        source_type="diagnosis",
        source_subtype="",
    )

    patient1 = Patient(
        patient_id=1,
        date_of_birth=datetime(1990, 1, 1),
    )
    patient1.add_events([e1, e2])

    patient2 = Patient(
        patient_id=2,
        date_of_birth=datetime(1993, 3, 1),
    )
    patient2.add_events([e1, e2, e2, e1])

    return [patient1, patient2]


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
    assert isinstance(inputs_ids["diagnosis"], torch.Tensor)
    assert isinstance(inputs_ids["age"], torch.Tensor)
    assert isinstance(inputs_ids["segment"], torch.Tensor)
    assert isinstance(inputs_ids["position"], torch.Tensor)

    # forward
    outputs = embedding_module(inputs_ids)


def test_masking_fn(patients: list):
    """
    Test masking function
    """
    emb = BEHRTEmbedder(d_model=384, dropout_prob=0.1, max_sequence_length=128)
    encoder_layer = nn.TransformerEncoderLayer(d_model=384, nhead=6)
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    emb.fit(patients)

    task = BEHRTMaskingTask(embedding_module=emb, encoder_module=encoder)

    inputs_ids = emb.collate_fn(patients)

    masked_input_ids, masked_labels = task.masking_fn(inputs_ids)

    # assert types
    assert isinstance(masked_input_ids, dict)
    assert isinstance(masked_input_ids["diagnosis"], torch.Tensor)
    assert isinstance(masked_labels, torch.Tensor)

    # assert that the masked labels are same as the input ids where they are not masked? # TODO

    # assert that padding is ignored? # TODO


def test_main(patients: list, tmp_path: Path):
    """
    Tests the general intended workflow
    """
    emb = BEHRTEmbedder(
        d_model=384, dropout_prob=0.1, max_sequence_length=128
    )  # probably some more args here    # TODO
    encoder_layer = nn.TransformerEncoderLayer(d_model=384, nhead=6)
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
    task = BEHRTMaskingTask(  # TODO
        embedding_module=emb, encoder_module=encoder
    )  # this includes the loss and the MLM head
    # ^should masking be here?

    optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)

    train_dataset = PatientDataset(train_patients)  # TODO
    val_dataset = PatientDataset(val_patients)

    # chain two functions:
    #     task.collate_fn,# handles masking
    #     emb.collate_fn, # handles padding, indexing etc.
    collate_fn = lambda x: task.masking_fn(emb.collate_fn(x))

    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
    )

    emb.fit(train_patients, add_mask_token=True)

    trainer = Trainer(task, optimizer, train_dataloader, val_dataloader)  # TODO
    trainer.train(steps=20)
    trainer.evaluate()

    # test that is can be loaded and saved from disk
    trainer.save_to_disk(tmp_path)
    trainer.load_from_disk(tmp_path)

    # tes that it can log data
    trainer.log({"step": 1, "loss": 0.1})
