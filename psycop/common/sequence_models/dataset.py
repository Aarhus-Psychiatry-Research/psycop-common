"""
Defines the dataset class for patient data
"""

from typing import Sequence
from torch.utils.data import Dataset

from psycop.common.data_structures import Patient
from psycop.common.data_structures.prediction_time import PredictionTime


class PatientDataset(Dataset):
    def __init__(self, patients: list[Patient]) -> None:
        self.patients: list[Patient] = patients

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> Patient:
        return self.patients[idx]


class PredictionTimeDataset(Dataset):
    def __init__(self, prediction_times: Sequence[PredictionTime]) -> None:
        self.prediction_times = prediction_times

    def __len__(self) -> int:
        return len(self.prediction_times)

    def __getitem__(self, idx: int) -> PredictionTime:
        return self.prediction_times[
            idx
        ]  # Uncertian exactly what the sequence should look like here; we probably want to construct the sequence in the embedder, so this just returns PredictionTimes?
