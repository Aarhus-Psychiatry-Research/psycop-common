import datetime as dt
from typing import Sequence

import polars as pl

from psycop.common.data_structures.static_feature import StaticFeature
from psycop.common.data_structures.temporal_event import TemporalEvent
from psycop.common.data_structures.test_patient import get_test_patient
from psycop.common.feature_generation.sequences.event_dataframes_to_patient import (
    EventDataFramesToPatients,
)
from psycop.common.test_utils.str_to_df import str_to_pl_df


def create_date_of_birth_df(patient_ids: Sequence[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "dw_ek_borger": patient_ids,
            "timestamp": [dt.datetime(year=1990, month=1, day=1) for _ in patient_ids],
        }
    )


def test_temporal_events():
    test_data = str_to_pl_df(
        """dw_ek_borger,timestamp,source,value
1,2020-01-01 00:00:00,source1,0
1,2020-01-01 00:00:00,source1,1
2,2020-01-01 00:00:00,source1,2
                             """,
    )

    patient_1 = get_test_patient(
        patient_id=1,
    )
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
        ],
    )

    patient_2 = get_test_patient(
        patient_id=2,
    )
    patient_2.add_events(
        [
            TemporalEvent(
                timestamp=dt.datetime(year=2020, month=1, day=1),
                value=2,
                source_type="source1",
                source_subtype=None,
            ),
        ],
    )
    expected_patients = [patient_1, patient_2]

    unpacked = EventDataFramesToPatients().unpack(
        source_event_dataframes=[test_data],
        date_of_birth_df=create_date_of_birth_df(patient_ids=[1, 2]),
    )
    assert unpacked == expected_patients


def test_static_features():
    test_data = str_to_pl_df(
        """dw_ek_borger,source,value
1,test,0
                             """,
    )

    expected_patient = get_test_patient(patient_id=1)

    expected_patient.add_events(
        [StaticFeature(source_type="test", value=0)],
    )

    unpacked = EventDataFramesToPatients().unpack(
        source_event_dataframes=[test_data],
        date_of_birth_df=create_date_of_birth_df(patient_ids=[1]),
    )

    assert unpacked == [expected_patient]


def test_multiple_event_sources():
    test_data = str_to_pl_df(
        """dw_ek_borger,source,value
1,test,0
                             """,
    )
    expected_static_event = StaticFeature(source_type="test", value=0)

    test_data2 = str_to_pl_df(
        """dw_ek_borger,source,timestamp,value
1,test2,2023-01-01,1
                             """,
    )
    expected_temporal_event = TemporalEvent(
        source_type="test2",
        source_subtype=None,
        timestamp=dt.datetime(2023, 1, 1),
        value=1,
    )

    expected_patient = get_test_patient(patient_id=1)
    expected_patient.add_events(
        [expected_static_event, expected_temporal_event],
    )

    unpacked = EventDataFramesToPatients().unpack(
        source_event_dataframes=[test_data, test_data2],
        date_of_birth_df=create_date_of_birth_df(patient_ids=[1]),
    )

    assert unpacked == [expected_patient]
