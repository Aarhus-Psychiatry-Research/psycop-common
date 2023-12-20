import datetime as dt
from pathlib import Path

import pytest

from psycop.common.data_structures.patient import Patient, PatientSlice
from psycop.common.data_structures.prediction_time import PredictionTime
from psycop.common.data_structures.temporal_event import TemporalEvent
from psycop.common.feature_generation.sequences.patient_slice_getter import (
    BasePredictionTimeCreator,
)
from psycop.common.sequence_models.dataset import PredictionTimeDataset
from psycop.common.sequence_models.registry import Registry
from psycop.common.sequence_models.train import train

from .test_encoder_for_clf import arm_within_docker


@Registry.datasets.register("fake_patient_slices_with_labels")
class FakeLabelledPatientSlices(BasePredictionTimeCreator):
    def __init__(self):
        pass

    def get_dataset(self) -> PredictionTimeDataset:
        temporal_events = [
            TemporalEvent(
                timestamp=dt.datetime.now(),
                source_type="diagnosis",
                source_subtype="fake_subtype",
                value="fake_value",
            ),
        ]

        patient_slices = PredictionTimeDataset(
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


@pytest.mark.skipif(
    arm_within_docker(),
    reason="Skipping test on ARM within docker. Some tests fail, unknown reason, see https://github.com/Aarhus-Psychiatry-Research/psycop-common/issues/348",
)
def test_finetune():
    config_path = Path(__file__).parent / "test_finetuning.cfg"
    train(config_path)
