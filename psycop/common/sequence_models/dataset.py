"""
Defines the dataset class for patient data
"""

from torch.utils.data import Dataset

from psycop.common.data_structures import Patient


class PatientDataset(Dataset):
    def __init__(self, patients: list[Patient]) -> None:
        self.patients: list[Patient] = patients

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> Patient:
        return self.patients[idx]


class PatientDatasetWithLabels(Dataset):
    def __init__(self, patients: list[Patient], labels: list[int]) -> None:
        self.patients: list[Patient] = patients
        self.labels: list[int] = labels
        assert len(self.patients) == len(self.labels)

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> tuple[Patient, int]:
        return self.patients[idx], self.labels[idx]
