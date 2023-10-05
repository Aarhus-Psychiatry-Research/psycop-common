import polars as pl
from psycop.common.feature_generation.sequences.map_diagnoses_to_caliber import (
    map_icd10_to_caliber_categories,
)

from psycop.common.feature_generation.sequences.patient_loaders import (
    DiagnosisLoader,
)
from psycop.common.test_utils.str_to_df import str_to_pl_df


def test_diagnosis_formatting():
    df = str_to_pl_df(
        """dw_ek_borger,datotid_slut,diagnosegruppestreng
    1,2023-01-01,A:DF431                    # Keep all, even though doesn't end with a hashtag
    1,2023-01-01,A:DF431#+:ALFC3#B:DF329    # Keep up until, but not including, first hashtag
    """,
    )

    formatted_df = DiagnosisLoader().format_diagnosis_columns(df=df.lazy())

    resulting_diagnoses = formatted_df.collect().get_column("value").to_list()
    assert resulting_diagnoses == ["F431", "F431", "ALFC3", "F329"]
    types = formatted_df.collect().get_column("type").to_list()
    assert types == ["A", "A", "+", "B"]


def test_map_icd10_to_caliber_categories():
    df = str_to_pl_df(
        """dw_ek_borger,datotid_slut,diagnosegruppestreng
    1,2023-01-01,A:DF30     # Same caliber category as F31
    1,2023-01-01,A:DF31     # Same caliber category as F30
    1,2023-01-01,A:DF32 
    1,2023-01-01,A:DA00     # Add only one caliber category, though A00 belongs to two caliber categories
    1,2023-01-01,A:DF431    # Does not exist in caliber
    """,
    )

    formatted_df = DiagnosisLoader().format_diagnosis_columns(df=df.lazy())

    mapping_df = pl.read_csv(
        "psycop/common/feature_generation/sequences/caliber-icd10-mapping.csv"
    ).rename({"ICD10code": "value"})

    resulting_diagnoses = (
        map_icd10_to_caliber_categories(formatted_df, mapping_df)
        .collect()
        .get_column("value")
        .to_list()
    )

    assert resulting_diagnoses == [
        "Bipolar affective disorder and mania",
        "Bipolar affective disorder and mania",
        "Depression",
        "Bacterial Diseases (excl TB)",
    ]
