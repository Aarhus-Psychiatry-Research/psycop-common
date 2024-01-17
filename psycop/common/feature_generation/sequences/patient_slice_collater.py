from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from psycop.common.feature_generation.sequences.event_loader import DiagnosisLoader
from psycop.common.feature_generation.sequences.patient_loader import PatientLoader
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_split import (
    RegionalFilter,
)
from psycop.common.sequence_models.dataset import PatientSliceDataset
from psycop.common.sequence_models.registry import SequenceRegistry


@runtime_checkable
class BasePatientSliceCollater(Protocol):
    def get_dataset(self) -> PatientSliceDataset:
        ...


@SequenceRegistry.datasets.register("unlabelled_slice_creator")
@dataclass(frozen=True)
class PatientSliceCollater(BasePatientSliceCollater):
    patient_loader: PatientLoader

    def get_dataset(self) -> PatientSliceDataset:
        return PatientSliceDataset(
            [patient.as_slice() for patient in self.patient_loader.get_patients()],
        )


if __name__ == "__main__":
    patient_slices = PatientSliceCollater(
        patient_loader=PatientLoader(
            event_loaders=[DiagnosisLoader()],
            split_filter=RegionalFilter(splits_to_keep=["train"]),
            fraction=0.05,
        ),
    ).get_dataset()
