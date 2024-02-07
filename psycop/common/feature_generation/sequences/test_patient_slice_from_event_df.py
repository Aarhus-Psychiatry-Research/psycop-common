import datetime as dt

import pytest

from psycop.common.data_structures.static_feature import StaticFeature
from psycop.common.data_structures.temporal_event import TemporalEvent
from psycop.common.data_structures.test_patient import get_test_patient
from psycop.common.feature_generation.sequences.event_loader import DiagnosisLoader
from psycop.common.feature_generation.sequences.patient_slice_from_events import (
    PatientSliceColumnNames,
    PatientSliceFromEvents,
)
from psycop.common.feature_generation.sequences.utils_for_testing import get_test_date_of_birth_df
from psycop.common.test_utils.str_to_df import str_to_pl_df


def test_temporal_events():
    test_data = str_to_pl_df(
        """dw_ek_borger,timestamp,source,value
1,2020-01-01 00:00:00,source1,0
1,2020-01-01 00:00:00,source1,1
2,2020-01-01 00:00:00,source1,2
                             """
    )

    patient_1 = get_test_patient(patient_id=1)
    patient_1.add_events(
        [
            TemporalEvent(
                timestamp=dt.datetime(year=2020, month=1, day=1),
                value=0,
                source_type="source1",
                source_subtype=None,
            ),
            TemporalEvent(
                timestamp=dt.datetime(year=2020, month=1, day=1),
                value=1,
                source_type="source1",
                source_subtype=None,
            ),
        ]
    )

    patient_2 = get_test_patient(patient_id=2)
    patient_2.add_events(
        [
            TemporalEvent(
                timestamp=dt.datetime(year=2020, month=1, day=1),
                value=2,
                source_type="source1",
                source_subtype=None,
            )
        ]
    )
    expected_patients = [patient_1, patient_2]

    unpacked = PatientSliceFromEvents(
        column_names=PatientSliceColumnNames(source_subtype_col_name=None)
    ).unpack(
        source_event_dataframes=[test_data],
        date_of_birth_df=get_test_date_of_birth_df(patient_ids=[1, 2]),
    )
    assert unpacked == expected_patients


def test_static_features():
    test_data = str_to_pl_df(
        """dw_ek_borger,source,value
1,test,0
                             """
    )

    expected_patient = get_test_patient(patient_id=1)

    expected_patient.add_events([StaticFeature(source_type="test", value=0)])

    unpacked = PatientSliceFromEvents().unpack(
        source_event_dataframes=[test_data],
        date_of_birth_df=get_test_date_of_birth_df(patient_ids=[1]),
    )

    assert unpacked == [expected_patient]


def test_multiple_event_sources():
    test_data = str_to_pl_df(
        """dw_ek_borger,source,value
1,test,0
                             """
    )
    expected_static_event = StaticFeature(source_type="test", value=0)

    test_data2 = str_to_pl_df(
        """dw_ek_borger,source,timestamp,value
1,test2,2023-01-01,1
                             """
    )
    expected_temporal_event = TemporalEvent(
        source_type="test2", source_subtype=None, timestamp=dt.datetime(2023, 1, 1), value=1
    )

    expected_patient = get_test_patient(patient_id=1)
    expected_patient.add_events([expected_static_event, expected_temporal_event])

    unpacked = PatientSliceFromEvents(
        column_names=PatientSliceColumnNames(source_subtype_col_name=None)
    ).unpack(
        source_event_dataframes=[test_data, test_data2],
        date_of_birth_df=get_test_date_of_birth_df(patient_ids=[1]),
    )

    assert unpacked == [expected_patient]


def test_patient_without_date_of_birth_raises_error():
    test_data = str_to_pl_df(
        """dw_ek_borger,source,value
1,test,0
                             """
    )

    with pytest.raises(KeyError):
        PatientSliceFromEvents().unpack(
            source_event_dataframes=[test_data],
            date_of_birth_df=get_test_date_of_birth_df(patient_ids=[2]),
        )


def test_passing_patient_colnames():
    """Test that column names can be passed when unpacking events."""

    df = str_to_pl_df(
        """dw_ek_borger,datotid_slut,diagnosegruppestreng
    1,2023-01-01,A:DF431
    1,2023-01-01,A:DF431#+:ALFC3#B:DF329
    2,2020-01-01,A:DF431#+:ALFC3#B:DF329
    """
    )

    formatted_df = DiagnosisLoader().preprocess_diagnosis_columns(df=df.lazy()).collect()

    unpacked_with_source_subtype_column = PatientSliceFromEvents(
        column_names=PatientSliceColumnNames(source_subtype_col_name="type")
    ).unpack(
        source_event_dataframes=[formatted_df],
        date_of_birth_df=get_test_date_of_birth_df(patient_ids=[1, 2]),
    )

    unpacked_without_source_subtype_column = PatientSliceFromEvents(
        column_names=PatientSliceColumnNames(source_subtype_col_name=None)
    ).unpack(
        source_event_dataframes=[formatted_df],
        date_of_birth_df=get_test_date_of_birth_df(patient_ids=[1, 2]),
    )

    # Assert that source_subtypes are str when source_subtype_col_name is specified
    assert all(
        isinstance(event.source_subtype, str)
        for event in [e for p in unpacked_with_source_subtype_column for e in p.temporal_events]
    )
    # Assert that source_subtypes are None when source_subtype_col_name is not specified
    assert all(
        event.source_subtype is None
        for event in [e for p in unpacked_without_source_subtype_column for e in p.temporal_events]
    )
