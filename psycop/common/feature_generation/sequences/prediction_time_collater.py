import datetime as dt
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ...cohort_definition import CohortDefiner
from ...model_training_v2.trainer.preprocessing.step import PresplitStep
from ...sequence_models.dataset import PredictionTimeDataset
from ...sequence_models.registry import Registry
from .patient_loader import PatientLoader
from .prediction_times_from_cohort import PredictionTimesFromCohort


@runtime_checkable
class BasePredictionTimeCollater(Protocol):
    def get_dataset(self) -> PredictionTimeDataset:
        ...


@Registry.datasets.register("prediction_time_collater")
@dataclass(frozen=True)
class PredictionTimeCollater(BasePredictionTimeCollater):
    patient_loader: PatientLoader
    cohort_definer: CohortDefiner
    split_filter: PresplitStep
    lookbehind_days: int
    lookahead_days: int

    def get_dataset(
        self,
    ) -> PredictionTimeDataset:
        prediction_times = PredictionTimesFromCohort(
            cohort_definer=self.cohort_definer,
            patients=self.patient_loader.get_patients(),
            split_filter=self.split_filter,
        ).create_prediction_times(
            lookbehind=dt.timedelta(days=self.lookbehind_days),
            lookahead=dt.timedelta(days=self.lookahead_days),
        )

        return PredictionTimeDataset(prediction_times)
<<<<<<< Updated upstream
=======


if __name__ == "__main__":
    prediction_times = PredictionTimeCollater(
        PatientLoader(
            event_loaders=[DiagnosisLoader()],
            split_filter=RegionalFilter(splits_to_keep=["val"]),
            fraction=0.05,
        ),
        cohort_definer=T2DCohortDefiner(),
        lookahead_days=730,
        lookbehind_days=730,
    ).get_dataset()
    pass
>>>>>>> Stashed changes
