import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal

from psycop.common.feature_generation.sequences.patient_loaders import (
    DiagnosisLoader,
    keep_if_min_n_visits,
)
from psycop.common.test_utils.str_to_df import str_to_pl_df


def test_diagnosis_preprocessing():
    df = str_to_pl_df(
        """dw_ek_borger,datotid_slut,diagnosegruppestreng
    1,2023-01-01,A:DF431                    # Keep all, even though doesn't end with a hashtag
    1,2023-01-01,A:DF431                    # Duplicate, do not keep
    2,2023-01-01,A:DF439#+:ALFC3#B:DF329    # Keep up until, but not including, first hashtag
    """,
    )

    formatted_df = DiagnosisLoader().preprocess_diagnosis_columns(df=df.lazy())

    resulting_diagnoses = formatted_df.collect().get_column("value").to_list()
    assert resulting_diagnoses == ["F431", "F439", "ALFC3", "F329"]

    diagnosis_is_alphanumeric = [
        diagnosis.isalnum() for diagnosis in resulting_diagnoses
    ]  # Ensure no special characters remain, e.g., # or :
    pd.Series(diagnosis_is_alphanumeric).all()

    types = formatted_df.collect().get_column("type").to_list()
    assert types == ["A", "A", "+", "B"]


test_diagnosis_preprocessing()


def test_keep_min_n_visits():
    df = str_to_pl_df(
        """dw_ek_borger,timestamp,type,value,source
    1,2023-01-01,A,F431,diagnosis
    1,2023-01-01,B,F439,diagnosis
    1,2023-01-01,H,S610,diagnosis
    1,2023-01-01,+,F439,diagnosis           # Discard, since all diagnoses are from the same visit
    2,2023-01-01,A,F439,diagnosis
    2,2023-01-02,A,F439,diagnosis
    2,2023-01-03,A,F439,diagnosis
    2,2023-01-04,A,F439,diagnosis           # Keep, visits = 4
    """,
    )

    filtered_df = keep_if_min_n_visits(df.lazy(), n_visits=4)

    assert_frame_equal(df.filter(pl.col("dw_ek_borger") == 2), filtered_df.collect())
