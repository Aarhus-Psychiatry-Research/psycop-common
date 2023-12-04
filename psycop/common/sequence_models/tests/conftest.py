from collections.abc import Sequence

import pytest

from psycop.common.data_structures.patient import Patient, PatientSlice
from psycop.common.sequence_models import PatientSliceDataset
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
def optimizer_fn() -> OptimizerFn:
    return create_adamw(lr=0.03)


@pytest.fixture()
def lr_scheduler_fn() -> LRSchedulerFn:
    return create_linear_schedule_with_warmup(num_warmup_steps=2, num_training_steps=10)
