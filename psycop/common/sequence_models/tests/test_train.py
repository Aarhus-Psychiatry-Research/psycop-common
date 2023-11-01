from pathlib import Path

import pytest

from psycop.common.sequence_models.dataset import PatientSliceDataset
from psycop.common.sequence_models.registry import Registry
from psycop.common.sequence_models.train import train

from .utils import create_patients


@pytest.fixture()
def config_path() -> Path:
    return Path(__file__).parent / "test_config.cfg"


@Registry.datasets.register("test_dataset")
def create_test_dataset() -> PatientSliceDataset:
    patients = create_patients()
    return PatientSliceDataset(patient_slices=[p.as_slice() for p in patients])


def test_train(
    config_path: Path,
):  # TODO: this test fails, had to remove it to merge the other branch, added it again here but needs to be debugged
    """test of the train function"""
    train(config_path)
