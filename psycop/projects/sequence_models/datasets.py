"""
datasets for training and evaluation of sequence models
"""

import datetime as dt

from psycop.common.feature_generation.loaders.raw.load_ids import SplitName
from psycop.common.feature_generation.sequences.cohort_definer_to_prediction_times import \
    CohortToPredictionTimes
from psycop.common.feature_generation.sequences.patient_loaders import (
    DiagnosisLoader, PatientLoader)
from psycop.common.sequence_models.dataset import (PatientSliceDataset,
                                                   PatientSlicesWithLabels)
from psycop.common.sequence_models.registry import Registry
from psycop.projects.t2d.feature_generation.cohort_definition.t2d_cohort_definer import \
    T2DCohortDefiner


@Registry.datasets.register("patient_slice_dataset")
def create_patient_slice_dataset(
    min_n_visits: int,
    split_name: str,  # TODO: @martin hvordan ville du type hinte det her?
) -> PatientSliceDataset:
    train_patients = PatientLoader.get_split(
        event_loaders=[
            DiagnosisLoader(min_n_visits=min_n_visits),
        ],
        split=SplitName(split_name),
    )
    return PatientSliceDataset([p.as_slice() for p in train_patients])


@Registry.datasets.register("patient_slices_with_labels_for_t2d")
def create_patient_slices_with_labels_for_t2d(
    min_n_visits: int,
    split_name: str,
    lookbehind_days: int = 365,
    lookahead_days: int = 365,
) -> PatientSlicesWithLabels:
    patients = PatientLoader.get_split(
        event_loaders=[DiagnosisLoader(min_n_visits=min_n_visits)],
        split=SplitName(split_name),
    )
    dt.timedelta(days=365)

    prediction_times = CohortToPredictionTimes(
        cohort_definer=T2DCohortDefiner(),
        patients=patients,
    ).create_prediction_times(
        lookbehind=dt.timedelta(days=lookbehind_days),
        lookahead=dt.timedelta(days=lookahead_days),
    )

    return PatientSlicesWithLabels(prediction_times)
