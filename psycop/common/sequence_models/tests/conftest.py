from datetime import datetime
from typing import Sequence

import pytest
from torch import nn

from psycop.common.data_structures import TemporalEvent
from psycop.common.data_structures.patient import (
    Patient,
    PatientSlice,
    patients_to_infinite_slices,
)
from psycop.common.sequence_models import (
    BEHRTEmbedder,
    BEHRTForMaskedLM,
    PatientSliceDataset,
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
        source_subtype="A",
    )
    e2 = TemporalEvent(
        timestamp=datetime(2021, 1, 3),
        value="d2",
        source_type="diagnosis",
        source_subtype="A",
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
def patient_slices(patients: list[Patient]) -> Sequence[PatientSlice]:
    return patients_to_infinite_slices(patients)


@pytest.fixture()
def patient_dataset(patient_slices: list[PatientSlice]) -> PatientSliceDataset:
    return PatientSliceDataset(patient_slices=patient_slices)


@pytest.fixture()
def behrt_for_masked_lm(patients: list[PatientSlice]) -> BEHRTForMaskedLM:
    d_model = 32
    emb = BEHRTEmbedder(d_model=d_model, dropout_prob=0.1, max_sequence_length=128)
    emb.fit(patient_slices=patients, add_mask_token=True)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=int(d_model / 4),
        dim_feedforward=d_model * 4,
        batch_first=True,
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    # this includes the loss and the MLM head
    module = BEHRTForMaskedLM(
        embedding_module=emb,
        encoder_module=encoder,
        optimizer_kwargs={"lr": 1e-3},
        lr_scheduler_kwargs={"num_warmup_steps": 2, "num_training_steps": 10},
    )
    return module
