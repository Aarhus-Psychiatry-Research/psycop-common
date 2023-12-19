import datetime as dt
from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from ...cohort_definition import CohortDefiner
from ...sequence_models.dataset import PatientSliceDataset, PatientSlicesWithLabels
from ...sequence_models.registry import Registry
from ..loaders.raw.load_ids import SplitName
from .cohort_definer_to_prediction_times import CohortToPredictionTimes
from .patient_loaders import EventDfLoader, PatientLoader


@runtime_checkable
class BaseUnlabelledSliceCreator(Protocol):
    def get_patient_slices(self) -> PatientSliceDataset:
        ...


@Registry.datasets.register("unlabelled_slice_creator")
class UnlabelledSliceCreator(BaseUnlabelledSliceCreator):
    def __init__(
        self,
        split_name: Literal["train", "val", "test"],
        cohort_definer: CohortDefiner,
        event_loaders: Sequence[EventDfLoader],
        min_n_events: int | None = None,
        load_fraction: float = 1.0,
    ):
        self.split_name = split_name
        self.patient_loader = PatientLoader()
        self.cohort_definer = cohort_definer
        self.event_loaders = event_loaders
        self.load_fraction = load_fraction
        self.min_n_events = min_n_events

    def get_patient_slices(
        self,
    ) -> PatientSliceDataset:
        patients = PatientLoader.get_split(
            event_loaders=self.event_loaders,
            split=SplitName(self.split_name),
            min_n_events=self.min_n_events,
            fraction=self.load_fraction,
        )

        return PatientSliceDataset([patient.as_slice() for patient in patients])


@runtime_checkable
class BaseLabelledSliceCreator(Protocol):
    def get_patient_slices(self) -> PatientSlicesWithLabels:
        ...


@Registry.datasets.register("labelled_patient_slices")
class LabelledPatientSliceCreator(BaseLabelledSliceCreator):
    def __init__(
        self,
        split_name: Literal["train", "val", "test"],
        lookbehind_days: int,
        lookahead_days: int,
        cohort_definer: CohortDefiner,
        event_loaders: Sequence[EventDfLoader],
        load_fraction: float = 1.0,
    ):
        self.split_name = split_name
        self.lookbehind_days = lookbehind_days
        self.lookahead_days = lookahead_days
        self.patient_loader = PatientLoader()
        self.cohort_definer = cohort_definer
        self.event_loaders = event_loaders
        self.load_fraction = load_fraction

    def get_patient_slices(
        self,
    ) -> PatientSlicesWithLabels:
        prediction_times = CohortToPredictionTimes(
            cohort_definer=self.cohort_definer,
            patients=self.patient_loader.get_split(
                event_loaders=self.event_loaders,
                split=SplitName(self.split_name),
                fraction=self.load_fraction,
            ),
        ).create_prediction_times(
            lookbehind=dt.timedelta(days=self.lookbehind_days),
            lookahead=dt.timedelta(days=self.lookahead_days),
        )

        return PatientSlicesWithLabels(prediction_times)
