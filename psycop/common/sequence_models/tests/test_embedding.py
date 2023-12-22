import datetime as dt
from collections.abc import Sequence

import pytest
import torch

from psycop.common.data_structures import TemporalEvent
from psycop.common.data_structures.patient import Patient, PatientSlice
from psycop.common.sequence_models.embedders.BEHRT_embedders import BEHRTEmbedder
from psycop.common.sequence_models.embedders.interface import PatientSliceEmbedder


@pytest.mark.parametrize(
    "embedder",
    [BEHRTEmbedder(d_model=384, dropout_prob=0.1, max_sequence_length=128)],
)
def test_embeddings(
    patient_slices: Sequence[PatientSlice],
    embedder: PatientSliceEmbedder,
):
    """
    Test embedding interface
    """
    embedder.fit(patient_slices)

    inputs_ids = embedder.collate_patient_slices(patient_slices)

    assert isinstance(inputs_ids, dict)
    assert isinstance(inputs_ids["diagnosis"], torch.Tensor)  # type: ignore
    assert isinstance(inputs_ids["age"], torch.Tensor)  # type: ignore
    assert isinstance(inputs_ids["segment"], torch.Tensor)
    assert isinstance(inputs_ids["position"], torch.Tensor)

    embedder.forward(inputs_ids)


@pytest.mark.parametrize(
    "embedder",
    [BEHRTEmbedder(d_model=384, dropout_prob=0.1, max_sequence_length=128)],
)
def test_diagnosis_mapping(
    embedder: BEHRTEmbedder,
):
    """
    Test mapping of diagnosis from ICD10 to caliber
    """
    # Check that diagnosis codes that are not in the mapping are excluded
    # (this patient has no diagnosis codes in the mapping)
    patient = Patient(
        patient_id=11,
        date_of_birth=dt.datetime(year=1990, month=1, day=1),
    )

    # Add temporal events to check that diagnosis codes are mapped correctly
    temporal_events = [
        TemporalEvent(
            timestamp=dt.datetime(2021, 1, 1),
            value="A00",
            source_type="diagnosis",
            source_subtype="A",
        ),
        # Check that two different ICD10 codes map to the same caliber code (A00, A30 -> Bacterial Diseases (excl TB))
        TemporalEvent(
            timestamp=dt.datetime(2021, 1, 3),
            value="A30",
            source_type="diagnosis",
            source_subtype="A",
        ),
        # Check that diagnoses with different subtype than A are excluded
        TemporalEvent(
            timestamp=dt.datetime(2021, 1, 3),
            value="A30",
            source_type="diagnosis",
            source_subtype="B",
        ),
        # Check that ICD10 code maps to different caliber code than those above (I65 -> Transient ischaemic attack)
        TemporalEvent(
            timestamp=dt.datetime(2021, 1, 3),
            value="I65",
            source_type="diagnosis",
            source_subtype="A",
        ),
        # Check that F909 is mapped to F90 (Hyperkinetic disorders) since F909 is not in the mapping
        TemporalEvent(
            timestamp=dt.datetime(2021, 1, 3),
            value="F909",
            source_type="diagnosis",
            source_subtype="A",
        ),
    ]

    patient.add_events(temporal_events)

    patient_events: list[tuple[Patient, TemporalEvent]] = [
        (p, e)
        for p in [patient]
        for e in filter(embedder.is_A_diagnosis, p.temporal_events)
    ]
    diagnosis_codes: list[str] = [e.value for p, e in patient_events]  # type: ignore

    # map diagnosis codes
    mapped_diagnosis_codes = [
        embedder.map_icd10_to_caliber(d)
        for d in diagnosis_codes
        if embedder.map_icd10_to_caliber(d)
    ]

    assert mapped_diagnosis_codes == [
        "Bacterial Diseases (excl TB)",
        "Bacterial Diseases (excl TB)",
        "Transient ischaemic attack",
        "Hyperkinetic disorders",
    ]


@pytest.mark.parametrize(
    "embedder",
    [BEHRTEmbedder(d_model=384, dropout_prob=0.1, max_sequence_length=128)],
)
def test_reformat_and_filter(
    embedder: BEHRTEmbedder,
):
    """
    Test mapping of diagnosis from ICD10 to caliber
    """
    # Check that diagnosis codes that are not in the mapping are excluded
    # (this patient has no diagnosis codes in the mapping)
    patient = Patient(
        patient_id=11,
        date_of_birth=dt.datetime(year=1990, month=1, day=1),
    )

    # Add temporal events to check that diagnosis codes are mapped correctly
    temporal_events = [
        TemporalEvent(
            timestamp=dt.datetime(2021, 1, 1),
            value="A00",
            source_type="diagnosis",
            source_subtype="A",
        ),
        # Check that two different ICD10 codes map to the same caliber code (A00, A30 -> Bacterial Diseases (excl TB))
        TemporalEvent(
            timestamp=dt.datetime(2021, 1, 3),
            value="A30",
            source_type="diagnosis",
            source_subtype="A",
        ),
        # Check that diagnoses with different subtype than A are excluded
        TemporalEvent(
            timestamp=dt.datetime(2021, 1, 3),
            value="A30",
            source_type="diagnosis",
            source_subtype="B",
        ),
        # Check that ICD10 code maps to different caliber code than those above (I65 -> Transient ischaemic attack)
        TemporalEvent(
            timestamp=dt.datetime(2021, 1, 3),
            value="I65",
            source_type="diagnosis",
            source_subtype="A",
        ),
        # Check that F909 is mapped to F90 (Hyperkinetic disorders) since F909 is not in the mapping
        TemporalEvent(
            timestamp=dt.datetime(2021, 1, 3),
            value="F909",
            source_type="diagnosis",
            source_subtype="A",
        ),
    ]

    patient.add_events(temporal_events)
    patient_slices_mapped = embedder.reformat(
        [patient.as_slice()],
    )
    diagnosis_codes = [
        e.value for ps in patient_slices_mapped for e in ps.temporal_events
    ]

    assert diagnosis_codes == [
        "Bacterial Diseases (excl TB)",
        "Bacterial Diseases (excl TB)",
        "Transient ischaemic attack",
        "Hyperkinetic disorders",
    ]
