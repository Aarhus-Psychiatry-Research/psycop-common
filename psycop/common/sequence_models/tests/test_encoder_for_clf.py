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
def patient_dataset_with_labels(patients: list) -> PatientDatasetWithLabels:
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
    patient_dataset_with_labels: PatientDatasetWithLabels,
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
        patient_dataset_with_labels,
        batch_size=32,
        shuffle=True,
        collate_fn=clf.collate_fn,  # type: ignore
    )

    for input_ids, masked_labels in dataloader:
        output = clf(input_ids, masked_labels)
        loss = output["loss"]
        loss.backward()  # ensure that the backward pass works
