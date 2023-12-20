import polars as pl
from polars.testing import assert_frame_equal

from psycop.common.feature_generation.sequences.event_loader import DiagnosisLoader
from psycop.common.feature_generation.sequences.patient_loader import (
    keep_if_min_n_visits,
)
from psycop.common.test_utils.str_to_df import str_to_pl_df


def test_diagnosis_preprocessing():
    df = str_to_pl_df(
        """dw_ek_borger,datotid_slut,diagnosegruppestreng
    1,2023-01-01,A:DF431                    # Keep all, even though doesn't end with a hashtag
    1,2023-01-01,A:DF431                    # Duplicate, do not keep
    2,2023-01-01,A:DF439#+:ALFC3#B:DF329    # Extract diagnoses correctly
    5,2023-01-01,A:DF431#HO:DF431           # Keep first, exclude invalid prefix
    """,
    )

    formatted_df = DiagnosisLoader().preprocess_diagnosis_columns(
        df=df.lazy(),
    )

    resulting_diagnoses = formatted_df.collect().get_column("value").to_list()
    assert resulting_diagnoses == ["F431", "F439", "ALFC3", "F329", "F431"]

    diagnoses_are_alphanumeric = [
        diagnosis.isalnum() for diagnosis in resulting_diagnoses
    ]  # Ensure no special characters remain, e.g., # or :
    assert all(diagnoses_are_alphanumeric)

    types = formatted_df.collect().get_column("type").to_list()
    assert types == ["A", "A", "+", "B", "A"]


def test_keep_min_n_visits():
    df = str_to_pl_df(
        """dw_ek_borger,timestamp
    1,2023-01-01
    1,2023-01-01
    1,2023-01-01
    1,2023-01-01               # Discard, since all diagnoses are from the same visit
    2,2023-01-01
    2,2023-01-02
    2,2023-01-03
    2,2023-01-04               # Keep, visits = 4
    """,
    )

    filtered_df = keep_if_min_n_visits(df.lazy(), n_visits=4)

    assert_frame_equal(df.filter(pl.col("dw_ek_borger") == 2), filtered_df.collect())
