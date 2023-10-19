"""
Defines the dataset class for patient data
"""

from collections.abc import Sequence

from torch.utils.data import Dataset

from psycop.common.data_structures.patient import Patient, PatientSlice
from psycop.common.data_structures.prediction_time import PredictionTime


class PatientSliceDataset(Dataset[PatientSlice]):
    def __init__(self, patient_slices: Sequence[PatientSlice]) -> None:
        self.patient_slices: Sequence[PatientSlice] = patient_slices

    def __len__(self) -> int:
        return len(self.patient_slices)

    def __getitem__(self, idx: int) -> PatientSlice:
        return self.patient_slices[idx]


class PatientSlicesWithLabels(Dataset[tuple[PatientSlice, int]]):
    def __init__(self, prediction_times: Sequence[PredictionTime]) -> None:
        self.prediction_times = prediction_times

    def __len__(self) -> int:
        return len(self.prediction_times)

    def __getitem__(self, idx: int) -> tuple[PatientSlice, int]:
        pred_time = self.prediction_times[idx]
        label = int(pred_time.outcome)

        return (pred_time.patient_slice, label)
