import datetime as dt
from pathlib import Path
from typing import Literal

from psycop.common.data_structures.patient import Patient, PatientSlice
from psycop.common.data_structures.prediction_time import PredictionTime
from psycop.common.data_structures.temporal_event import TemporalEvent
from psycop.common.feature_generation.loaders.raw.load_ids import SplitName
from psycop.common.feature_generation.sequences.cohort_definer_to_prediction_times import (
    CohortToPredictionTimes,
)
from psycop.common.feature_generation.sequences.patient_loaders import (
    DiagnosisLoader,
    PatientLoader,
)
from psycop.common.sequence_models.dataset import PatientSlicesWithLabels
from psycop.common.sequence_models.registry import Registry
from psycop.common.sequence_models.tasks import BEHRTForMaskedLM
from psycop.common.sequence_models.train import train
from psycop.projects.t2d.feature_generation.cohort_definition.t2d_cohort_definer import (
    T2DCohortDefiner,
)


@Registry.datasets.register("fake_patient_slices_with_labels")
def create_fake_patient_slices_with_labels() -> PatientSlicesWithLabels:
    temporal_events = [
        TemporalEvent(
            timestamp=dt.datetime.now(),
            source_type="diagnosis",
            source_subtype="fake_subtype",
            value="fake_value",
        ),
    ]

    patient_slices = PatientSlicesWithLabels(
        prediction_times=[
            PredictionTime(
                prediction_timestamp=dt.datetime.now(),
                patient_slice=PatientSlice(
                    patient=Patient(
                        patient_id=1234,
                        date_of_birth=dt.datetime.now(),
                        unsorted_temporal_events=temporal_events,
                    ),
                    temporal_events=temporal_events,
                ),
                outcome=True,
            ),
        ],
    )

    return patient_slices


def test_finetune():
    config_path = Path(__file__).parent / "test_finetuning.cfg"
    train(config_path)
