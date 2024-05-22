import datetime as dt
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from psycop.common.cohort_definition import CohortDefiner
from psycop.common.feature_generation.sequences.event_loader import DiagnosisLoader
from psycop.common.feature_generation.sequences.patient_loader import PatientLoader
from psycop.common.feature_generation.sequences.prediction_times_from_cohort import (
    PredictionTimesFromCohort,
)
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_split import (
    RegionalFilter,
)
from psycop.common.sequence_models.dataset import PredictionTimeDataset
from psycop.common.sequence_models.registry import SequenceRegistry
from psycop.projects.t2d.feature_generation.cohort_definition.t2d_cohort_definer import (
    T2DCohortDefiner,
)


@runtime_checkable
class BasePredictionTimeCollater(Protocol):
    def get_dataset(self) -> PredictionTimeDataset: ...


@SequenceRegistry.datasets.register("prediction_time_collater")
@dataclass(frozen=True)
class PredictionTimeCollater(BasePredictionTimeCollater):
    patient_loader: PatientLoader
    cohort_definer: CohortDefiner
    lookbehind_days: int
    lookahead_days: int

    def get_dataset(self) -> PredictionTimeDataset:
        prediction_times = PredictionTimesFromCohort(
            cohort_definer=self.cohort_definer, patients=self.patient_loader.get_patients()
        ).create_prediction_times(
            lookbehind=dt.timedelta(days=self.lookbehind_days),
            lookahead=dt.timedelta(days=self.lookahead_days),
        )

        return PredictionTimeDataset(prediction_times)


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
