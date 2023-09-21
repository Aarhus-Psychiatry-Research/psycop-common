from copy import deepcopy
from datetime import datetime
from pathlib import Path
from statistics import mean

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from psycop.common.data_structures import Patient, TemporalEvent
from psycop.common.sequence_models import (
    BEHRTEmbedder,
    BEHRTForMaskedLM,
    PatientDataset,
    Trainer,
)
from psycop.common.sequence_models.checkpoint_savers.save_to_disk import (
    CheckpointToDisk,
)
from psycop.common.sequence_models.loggers.base import Logger



def test_trainer(
    patients: list[Patient],
    tmp_path: Path,
    trainable_module: BEHRTForMaskedLM,
):
    """
    Tests the general intended workflow of the Trainer class
    """
    patients = patients * 10
    midpoint = int(len(patients) / 2)
    train_patients = patients[:midpoint]
    val_patients = patients[midpoint:]

    train_dataset = PatientDataset(train_patients)
    val_dataset = PatientDataset(val_patients)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=trainable_module.collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=trainable_module.collate_fn,
    )

    trainer = init_test_trainer(checkpoint_path=tmp_path)
    trainer.fit(
        n_steps=1,
        model=trainable_module,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        resume_from_latest_checkpoint=False,
    )

    # Check that model can resume training
    final_training_steps = 10
    resumed_trainer = init_test_trainer(checkpoint_path=tmp_path)
    resumed_trainer.fit(
        n_steps=final_training_steps,
        model=deepcopy(trainable_module),
        train_dataloader=deepcopy(train_dataloader),
        val_dataloader=deepcopy(val_dataloader),
        resume_from_latest_checkpoint=True,
    )
    assert resumed_trainer.train_step == final_training_steps

    # Check that model loss decreases over training time
    logger: DummyLogger = resumed_trainer.logger  # type: ignore
    metrics = logger.metrics
    first_three_losses = [metrics[i]["Training loss"] for i in range(0, 3)]
    last_three_losses = [metrics[i]["Training loss"] for i in range(-3, 0)]
    final_loss_smaller_than_initial_loss = mean(first_three_losses) > mean(
        last_three_losses,
    )
    assert final_loss_smaller_than_initial_loss
