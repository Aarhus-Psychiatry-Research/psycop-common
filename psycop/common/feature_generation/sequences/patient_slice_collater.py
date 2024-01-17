from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ...sequence_models.dataset import PatientSliceDataset
from ...sequence_models.registry import Registry
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
<<<<<<< Updated upstream
=======


if __name__ == "__main__":
    patient_slices = PatientSliceCollater(
        patient_loader=PatientLoader(
            event_loaders=[DiagnosisLoader()],
            split_filter=RegionalFilter(splits_to_keep=["train"]),
            fraction=0.05,
        ),
    ).get_dataset()
    pass
>>>>>>> Stashed changes
