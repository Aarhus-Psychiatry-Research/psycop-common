from pathlib import Path

import pytest

from psycop.common.sequence_models.dataset import PatientSliceDataset
from psycop.common.sequence_models.registry import SequenceRegistry
from psycop.common.sequence_models.train import train

from ...feature_generation.sequences.patient_slice_collater import BasePatientSliceCollater
from .utils import create_patients


@SequenceRegistry.datasets.register("test_dataset")
class FakeSliceCreator(BasePatientSliceCollater):
    def __init__(self):
        pass

    def get_dataset(self) -> PatientSliceDataset:
        return PatientSliceDataset(patient_slices=[p.as_slice() for p in create_patients()])


@pytest.fixture
def config_path() -> Path:
    return Path(__file__).parent / "test_train.cfg"


def test_train(config_path: Path):
    """test of the train function"""
    train(config_path)
