from datetime import datetime

import pytest
from torch import nn
from torch.utils.data import DataLoader

from psycop.common.data_structures import Patient, TemporalEvent
from psycop.common.sequence_models import (
    AggregationModule,
    AveragePooler,
    BEHRTEmbedder,
    EncoderForClassification,
    PatientDatasetWithLabels,
)


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
def patient_dataset(patients: list) -> PatientDatasetWithLabels:
    return PatientDatasetWithLabels(patients, labels=[0, 1])


@pytest.fixture()
def embedding_module(patients: list[Patient]) -> BEHRTEmbedder:
    d_model = 32
    emb = BEHRTEmbedder(d_model=d_model, dropout_prob=0.1, max_sequence_length=128)
    emb.fit(patients, add_mask_token=True)
    return emb


@pytest.fixture()
def encoder_module() -> nn.Module:
    d_model = 32
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=int(d_model / 4),
        dim_feedforward=d_model * 4,
        batch_first=True,
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
    return encoder


@pytest.fixture()
def aggregation_module() -> AveragePooler:
    """
    A mean pooling module
    """
    return AveragePooler()


def test_encoder_for_clf_(
    patient_dataset: PatientDatasetWithLabels,
    embedding_module: BEHRTEmbedder,
    encoder_module: nn.Module,
    aggregation_module: AggregationModule,
):
    clf = EncoderForClassification(
        embedding_module=embedding_module,
        encoder_module=encoder_module,
        aggregation_module=aggregation_module,
        num_classes=1,
        optimizer_kwargs={"lr": 1e-3},
        lr_scheduler_kwargs={"num_warmup_steps": 2, "num_training_steps": 10},
    )

    dataloader = DataLoader(
        patient_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=clf.collate_fn,  # type: ignore
    )

    for input_ids, masked_labels in dataloader:
        output = clf(input_ids, masked_labels)
        loss = output["loss"]
        loss.backward()  # ensure that the backward pass works
