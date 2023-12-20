from collections.abc import Sequence

import pytest
from torch import nn

from psycop.common.data_structures.patient import Patient, PatientSlice
from psycop.common.sequence_models import BEHRTEmbedder, PatientSliceDataset
from psycop.common.sequence_models.aggregators import Aggregator, CLSAggregator
from psycop.common.sequence_models.optimizers import (
    LRSchedulerFn,
    OptimizerFn,
    create_adamw,
    create_linear_schedule_with_warmup,
)

from .utils import create_patients


@pytest.fixture()
def patients() -> list[Patient]:
    """
    Returns a list of patient objects
    """
    return create_patients()


@pytest.fixture()
def patient_slices(patients: list[Patient]) -> Sequence[PatientSlice]:
    return [p.as_slice() for p in patients]


@pytest.fixture()
def patient_dataset(patient_slices: list[PatientSlice]) -> PatientSliceDataset:
    return PatientSliceDataset(patient_slices=patient_slices)


@pytest.fixture()
def optimizer() -> OptimizerFn:
    return create_adamw(lr=0.03)


@pytest.fixture()
def lr_scheduler_fn() -> LRSchedulerFn:
    return create_linear_schedule_with_warmup(num_warmup_steps=2, num_training_steps=10)


@pytest.fixture()
def embedder(patient_slices: list[PatientSlice]) -> BEHRTEmbedder:
    d_model = 32
    emb = BEHRTEmbedder(d_model=d_model, dropout_prob=0.1, max_sequence_length=128)
    emb.fit(patient_slices, add_mask_token=True)
    return emb


@pytest.fixture()
def encoder() -> nn.Module:
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
def aggregator() -> Aggregator:
    return CLSAggregator()
