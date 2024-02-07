"""
Defines the dataset class for patient data
"""

import logging
from collections.abc import Sequence
from typing import Callable

from torch.utils.data import Dataset
from tqdm import tqdm

from psycop.common.data_structures.patient import PatientSlice
from psycop.common.data_structures.prediction_time import PredictionTime

log = logging.getLogger(__name__)


class PatientSliceDataset(Dataset[PatientSlice]):
    def __init__(self, patient_slices: Sequence[PatientSlice]) -> None:
        self.patient_slices: Sequence[PatientSlice] = patient_slices

    def __len__(self) -> int:
        return len(self.patient_slices)

    def __getitem__(self, idx: int) -> PatientSlice:
        return self.patient_slices[idx]

    def filter_patients(
        self, filter_fn: Callable[[Sequence[PatientSlice]], Sequence[PatientSlice]]
    ) -> None:
        self.patient_slices = filter_fn(self.patient_slices)


class PredictionTimeDataset(Dataset[PredictionTime]):
    def __init__(self, prediction_times: Sequence[PredictionTime]) -> None:
        self.prediction_times = prediction_times

    def __len__(self) -> int:
        return len(self.prediction_times)

    def __getitem__(self, idx: int) -> PredictionTime:
        return self.prediction_times[idx]

    @property
    def patient_slices(self) -> Sequence[PatientSlice]:
        return [pred_time.patient_slice for pred_time in self.prediction_times]

    def filter_patients(
        self, filter_fn: Callable[[Sequence[PatientSlice]], Sequence[PatientSlice]]
    ) -> None:
        pred_times: list[PredictionTime] = []

        log.info("Filtering patients")
        for pred_time in tqdm(self.prediction_times):
            filtered_slice = filter_fn([pred_time.patient_slice])

            if len(filtered_slice) != 1:
                raise ValueError(
                    f"Filtering resulted in {len(filtered_slice)} patient slices. " "Expected 1."
                )

            new_pred_time = PredictionTime(
                prediction_timestamp=pred_time.prediction_timestamp,
                patient_slice=filtered_slice[0],
                outcome=pred_time.outcome,
            )
            pred_times.append(new_pred_time)

        self.prediction_times = pred_times
