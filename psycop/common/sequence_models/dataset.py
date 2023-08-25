"""
Defines the dataset class for patient data
"""

from torch.utils.data import Dataset

from psycop.common.data_structures import Patient


class PatientDataset(Dataset):
    def __init__(self, patients: list[Patient]):
        self.patients = patients

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx: int):
        return self.patients[idx]
