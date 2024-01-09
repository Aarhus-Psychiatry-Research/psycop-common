from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from ...cohort_definition import CohortDefiner
from ...sequence_models.dataset import PatientSliceDataset
from ...sequence_models.registry import Registry
from ..loaders.raw.load_ids import SplitName
from .event_loader import EventLoader
from .patient_loader import PatientLoader


@runtime_checkable
class BasePatientSliceCollater(Protocol):
    def get_dataset(self) -> PatientSliceDataset:
        ...


@Registry.datasets.register("unlabelled_slice_creator")
@dataclass(frozen=True)
class PatientSliceCollater(BasePatientSliceCollater):
    patient_loader: PatientLoader

    def get_dataset(self) -> PatientSliceDataset:
        return PatientSliceDataset(
            [patient.as_slice() for patient in self.patient_loader.get_patients()],
        )
