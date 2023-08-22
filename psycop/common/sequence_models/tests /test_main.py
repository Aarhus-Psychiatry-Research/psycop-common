from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from psycop.common.sequence_models import (
    BEHRTEmbedder,
    BEHRTMaskingTask,
    PatientDataset,
    Trainer,
)


@pytest.fixture
def patients() -> List[Patient]:
    """
    Returns a list of patient objects
    """
    return []


def test_main(patients: list, tmp_path: Path):
    """
    Tests the general intended workflow
    """
    emb = BEHRTEmbedder(d_model=384)  # probably some more args here    # TODO
    encoder_layer = nn.TransformerEncoderLayer(d_model=384, nhead=6)
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
    task = BEHRTMaskingTask(                                            # TODO
        embedding_module=emb, encoder_module=encoder
    )  # this includes the loss and the MLM head
    # ^should masking be here?

    optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)

    train_dataset = PatientDataset(train_patients)                      # TODO
    val_dataset = PatientDataset(val_patients)

    # chain two functions:
    #     task.collate_fn,# handles masking
    #     emb.collate_fn, # handles padding, indexing etc.
    collate_fn = lambda x: emb.collate_fn(task.masking_fn(x))

    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
    )

    emb.fit(train_patients)

    trainer = Trainer(task, optimizer, train_dataloader, val_dataloader)   # TODO
    trainer.train(steps=20)
    trainer.evaluate()

    # test that is can be loaded and saved from disk
    trainer.save_to_disk(tmp_path)
    trainer.load_from_disk(tmp_path)

    # tes that it can log data
    trainer.log({"step": 1, "loss": 0.1})
