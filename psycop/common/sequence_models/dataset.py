"""
Defines the dataset class for patient data
"""

from collections.abc import Sequence

from torch.utils.data import Dataset

from psycop.common.data_structures import Patient
from psycop.common.data_structures.patient_slice import PatientSlice
from psycop.common.data_structures.prediction_time import PredictionTime


class PatientDataset(Dataset):
    def __init__(self, patients: list[Patient]) -> None:
        self.patients: list[Patient] = patients

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> Patient:
        return self.patients[idx]


class PatientSlicesWithLabels(Dataset):
    def __init__(self, prediction_times: Sequence[PredictionTime]) -> None:
        self.prediction_times = prediction_times

    def __len__(self) -> int:
        return len(self.prediction_times)

    def __getitem__(self, idx: int) -> tuple[PatientSlice, int]:
        pred_time = self.prediction_times[idx]
        patient_slice = pred_time.to_patient_slice()
        label = int(pred_time.outcome)
        return (patient_slice, label)
