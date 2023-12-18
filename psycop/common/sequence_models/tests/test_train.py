from pathlib import Path

import pytest

from psycop.common.sequence_models.dataset import PatientSliceDataset
from psycop.common.sequence_models.registry import Registry
from psycop.common.sequence_models.train import train

from ...feature_generation.sequences.patient_slice_getter import (
    BaseUnlabelledSliceCreator,
)
from .utils import create_patients


@pytest.fixture()
def config_path() -> Path:
    return Path(__file__).parent / "test_config.cfg"


@Registry.datasets.register("test_dataset")
class FakeSliceCreator(BaseUnlabelledSliceCreator):
    def __init__(self):
        pass

    def get_patient_slices(self) -> PatientSliceDataset:
        return PatientSliceDataset(
            patient_slices=[p.as_slice() for p in create_patients()],
        )


def test_train(
    config_path: Path,
):
    """test of the train function"""
    train(config_path)
