import datetime as dt

import polars as pl

from psycop.common.cohort_definition import CohortDefiner, FilteredPredictionTimeBundle
from psycop.common.data_structures.test_patient import get_test_patient_slice
from psycop.common.feature_generation.sequences.cohort_definer_to_prediction_times import (
    CohortToPredictionTimes,
)
from psycop.common.test_utils.str_to_df import str_to_pl_df


class MockCohortDefiner(CohortDefiner):
    @staticmethod
    def get_filtered_prediction_times_bundle() -> FilteredPredictionTimeBundle:
        df = str_to_pl_df(
            """dw_ek_borger,timestamp
    1,2021-01-01
    2,2022-01-01
    """,
        )
        return FilteredPredictionTimeBundle(
            prediction_times=df,
            filter_steps=[],
        )

    @staticmethod
    def get_outcome_timestamps() -> pl.DataFrame:
        df = str_to_pl_df(
            """dw_ek_borger,timestamp
    1,2021-01-02
    """,
        )
        return df


def test_polars_dataframe_to_dict():
    """Test that each prediction time is mapped to the correct patient."""
    prediction_times = CohortToPredictionTimes(
        cohort_definer=MockCohortDefiner(),
        patient_objects=[
            get_test_patient_slice(patient_id=1),
            get_test_patient_slice(patient_id=2),
        ],
    ).create_prediction_times(
        lookbehind=dt.timedelta(days=1),
        lookahead=dt.timedelta(days=1),
    )

    assert len(prediction_times) == 2
    patient_1 = list(
        filter(lambda x: x.patient_slice.patient_id == 1, prediction_times)
    )[0]
    assert patient_1.prediction_timestamp == dt.datetime(2021, 1, 1)
    # The rest of the prediction time creation logic is tested in the patient object tests
