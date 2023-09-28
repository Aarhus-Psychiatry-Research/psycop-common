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


class DummyLogger(Logger):
    def __init__(self, project_name: str, run_name: str):
        self.metrics: list[dict[str, float]] = []
        self.run_name = run_name
        self.project_name = project_name

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics += [metrics]

    def log_hyperparams(self, params: dict[str, float | str]) -> None:
        pass


@pytest.fixture()
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


@pytest.fixture()
def patient_dataset(patients: list) -> PatientDataset:
    return PatientDataset(patients)


@pytest.fixture()
def trainable_module(patients: list[Patient]) -> BEHRTForMaskedLM:
    d_model = 32
    emb = BEHRTEmbedder(d_model=d_model, dropout_prob=0.1, max_sequence_length=128)
    emb.fit(patients=patients, add_mask_token=True)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=int(d_model / 4),
        dim_feedforward=d_model * 4,
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    # this includes the loss and the MLM head
    module = BEHRTForMaskedLM(
        embedding_module=emb,
        encoder_module=encoder,
    )
    return module


def test_behrt(patient_dataset: PatientDataset):
    d_model = 32
    emb = BEHRTEmbedder(d_model=d_model, dropout_prob=0.1, max_sequence_length=128)
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=int(d_model / 4),
        dim_feedforward=d_model * 4,
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    patients = patient_dataset.patients
    emb.fit(patients, add_mask_token=True)

    behrt = BEHRTForMaskedLM(embedding_module=emb, encoder_module=encoder)

    dataloader = DataLoader(
        patient_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=behrt.collate_fn,
    )

    for input_ids, masked_labels in dataloader:
        output = behrt(input_ids, masked_labels)
        loss = output["loss"]
        loss.backward()  # ensure that the backward pass works


def init_test_trainer(checkpoint_path: Path) -> Trainer:
    ckpt_saver = CheckpointToDisk(
        checkpoint_path=checkpoint_path,
        override_on_save=True,
    )
    logger = DummyLogger(run_name="test_run", project_name="test")
    return Trainer(
        device=torch.device("cpu"),
        validate_every_n_steps=1,
        n_samples_to_validate_on=2,
        logger=logger,
        checkpoint_savers=[ckpt_saver],
        save_every_n_steps=1,
    )


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
    first_three_losses = [metrics[i]["Training loss"] for i in range(0, 5)]
    last_three_losses = [metrics[i]["Training loss"] for i in range(-5, 0)]
    final_loss_smaller_than_initial_loss = mean(first_three_losses) > mean(
        last_three_losses,
    )
    assert final_loss_smaller_than_initial_loss
