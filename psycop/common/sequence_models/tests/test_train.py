from pathlib import Path

import pytest

from psycop.common.sequence_models.dataset import PatientSliceDataset
from psycop.common.sequence_models.registry import Registry

from .utils import create_patients


@pytest.fixture()
def config_path() -> Path:
    return Path(__file__).parent / "test_config.cfg"


@Registry.datasets.register("test_dataset")
def create_test_dataset() -> PatientSliceDataset:
    patients = create_patients()
    return PatientSliceDataset(patient_slices=[p.as_slice() for p in patients])
