from collections.abc import Sequence
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader

from psycop.common.data_structures.patient import PatientSlice
from psycop.common.sequence_models.dataset import PatientSliceDataset
from psycop.common.sequence_models.embedders.BEHRT_embedders import BEHRTEmbedder
from psycop.common.sequence_models.optimizers import (
    create_adamw,
    create_linear_schedule_with_warmup,
)

from ..tasks.pretrainer_behrt import PretrainerBEHRT
from .test_encoder_for_clf import TEST_CHECKPOINT_DIR


def test_behrt(patient_dataset: PatientSliceDataset):
    d_model = 32
    emb = BEHRTEmbedder(d_model=d_model, dropout_prob=0.1, max_sequence_length=128)
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=int(d_model / 4), dim_feedforward=d_model * 4, batch_first=True
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    patients = patient_dataset.patient_slices
    emb.fit(patient_slices=patients, add_mask_token=True)

    adam_fn = create_adamw(lr=0.03)
    lr_scheduler_fn = create_linear_schedule_with_warmup(num_warmup_steps=2, num_training_steps=10)

    behrt = PretrainerBEHRT(
        embedder=emb, encoder=encoder, optimizer=adam_fn, lr_scheduler=lr_scheduler_fn
    )

    dataloader = DataLoader(
        patient_dataset, batch_size=32, shuffle=True, collate_fn=behrt.collate_fn
    )

    trainer = pl.Trainer(max_epochs=1, accelerator="cpu")
    trainer.fit(behrt, train_dataloaders=dataloader)

    for input_ids, masked_labels in dataloader:
        output = behrt(input_ids, masked_labels)
        loss = output["loss"]
        loss.backward()  # ensure that the backward pass works


def create_behrt(
    patient_slices: Sequence[PatientSlice],
    d_model: int = 32,
    dropout_prob: float = 0.1,
    max_sequence_length: int = 128,
    n_heads: int = 8,
    dim_feedforward: int = 128,
    num_layers: int = 2,
) -> PretrainerBEHRT:
    """
    Creates a model for testing
    """

    emb = BEHRTEmbedder(
        d_model=d_model, dropout_prob=dropout_prob, max_sequence_length=max_sequence_length
    )
    emb.fit(patient_slices=patient_slices, add_mask_token=True)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    optimizer = create_adamw(lr=0.03)
    lr_scheduler_fn = create_linear_schedule_with_warmup(num_warmup_steps=2, num_training_steps=10)

    # this includes the loss and the MLM head
    module = PretrainerBEHRT(
        embedder=emb, encoder=encoder, optimizer=optimizer, lr_scheduler=lr_scheduler_fn
    )
    return module


def create_trainer(save_dir: Path) -> pl.Trainer:
    callbacks = [
        ModelCheckpoint(
            dirpath=save_dir / "checkpoints",
            every_n_epochs=1,
            verbose=True,
            save_top_k=1,
            mode="min",
            monitor="val_loss",
        )
    ]
    trainer = pl.Trainer(
        accelerator="cpu",
        val_check_interval=2,
        max_steps=5,
        callbacks=callbacks,  # type: ignore
    )

    return trainer


def test_module_with_trainer(patient_slices: Sequence[PatientSlice], tmp_path: Path):
    """
    Tests the general intended workflow of the Trainer class
    """
    override_test_checkpoints = False
    if override_test_checkpoints:
        tmp_path = TEST_CHECKPOINT_DIR.parent

    n_patients = 10
    more_patients = list(patient_slices) * n_patients
    midpoint = int(n_patients / 2)

    train_patients = more_patients[:midpoint]
    val_patients = more_patients[midpoint:]

    train_dataset = PatientSliceDataset(train_patients)
    val_dataset = PatientSliceDataset(val_patients)

    batch_size = 2

    trainable_module = create_behrt(patient_slices=train_patients)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=trainable_module.collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, collate_fn=trainable_module.collate_fn
    )

    trainer = create_trainer(save_dir=tmp_path)
    trainer.fit(
        model=trainable_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    # Checkpoints are saved
    checkpoint_paths = list((tmp_path / "checkpoints").glob("*.ckpt"))
    assert len(checkpoint_paths) >= 1

    # Checkpoint can be loaded
    # Note that load_from_checkpoint raises a FileNotFoundError if the checkpoint does not exist.
    # Hence, this would fail if we could not load the checkpoint.
    loaded_model = PretrainerBEHRT.load_from_checkpoint(checkpoint_paths[0])
    trainer.fit(model=loaded_model, train_dataloaders=train_dataloader)
