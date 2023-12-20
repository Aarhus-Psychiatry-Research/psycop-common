from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from ...cohort_definition import CohortDefiner
from ...sequence_models.dataset import PatientSliceDataset
from ...sequence_models.registry import Registry
from ..loaders.raw.load_ids import SplitName
from .event_loader import EventLoader
from .patient_loader import PatientLoader


@runtime_checkable
class BaseUnlabelledSliceCreator(Protocol):
    def get_dataset(self) -> PatientSliceDataset:
        ...


@Registry.datasets.register("unlabelled_slice_creator")
class UnlabelledSliceCollater(BaseUnlabelledSliceCreator):
    def __init__(
        self,
        split_name: Literal["train", "val", "test"],
        cohort_definer: CohortDefiner,
        event_loaders: Sequence[EventLoader],
        min_n_events: int | None = None,
        load_fraction: float = 1.0,
    ):
        self.split_name = split_name
        self.patient_loader = PatientLoader()
        self.cohort_definer = cohort_definer
        self.event_loaders = event_loaders
        self.load_fraction = load_fraction
        self.min_n_events = min_n_events

    def get_dataset(
        self,
    ) -> PatientSliceDataset:
        patients = PatientLoader.get_split(
            event_loaders=self.event_loaders,
            split=SplitName(self.split_name),
            min_n_events=self.min_n_events,
            fraction=self.load_fraction,
        )

        return PatientSliceDataset([patient.as_slice() for patient in patients])
