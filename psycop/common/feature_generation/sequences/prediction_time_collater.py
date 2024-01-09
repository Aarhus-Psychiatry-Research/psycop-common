import datetime as dt
from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from ...cohort_definition import CohortDefiner
from ...sequence_models.dataset import PredictionTimeDataset
from ...sequence_models.registry import Registry
from ..loaders.raw.load_ids import SplitName
from .event_loader import EventLoader
from .patient_loader import PatientLoader
from .prediction_times_from_cohort import PredictionTimesFromCohort


@runtime_checkable
class BasePredictionTimeCollater(Protocol):
    def get_dataset(self) -> PredictionTimeDataset:
        ...


@Registry.datasets.register("prediction_time_collater")
class PredictionTimeCollater(BasePredictionTimeCollater):
    def __init__(
        self,
        split_name: Literal["train", "val", "test"],
        lookbehind_days: int,
        lookahead_days: int,
        cohort_definer: CohortDefiner,
        event_loaders: Sequence[EventLoader],
        load_fraction: float = 1.0,
    ):
        self.split_name = split_name
        self.lookbehind_days = lookbehind_days
        self.lookahead_days = lookahead_days
        self.patient_loader = PatientLoader()
        self.cohort_definer = cohort_definer
        self.event_loaders = event_loaders
        self.load_fraction = load_fraction

    def get_dataset(
        self,
    ) -> PredictionTimeDataset:
        prediction_times = PredictionTimesFromCohort(
            cohort_definer=self.cohort_definer,
            patients=self.patient_loader.get_patients(
                event_loaders=self.event_loaders,
                split=SplitName(self.split_name),
                fraction=self.load_fraction,
            ),
        ).create_prediction_times(
            lookbehind=dt.timedelta(days=self.lookbehind_days),
            lookahead=dt.timedelta(days=self.lookahead_days),
        )

        return PredictionTimeDataset(prediction_times)
