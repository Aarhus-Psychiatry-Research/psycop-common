from psycop.common.feature_generation.sequences.patient_loaders import (
    DiagnosisLoader,
)
from psycop.common.test_utils.str_to_df import str_to_pl_df


def test_diagnosis_preproessing():
    df = str_to_pl_df(
        """dw_ek_borger,datotid_slut,diagnosegruppestreng
    1,2023-01-01,A:DF431                    # Keep all, even though doesn't end with a hashtag
    1,2023-01-01,A:DF431#+:ALFC3#B:DF329    # Keep up until, but not including, first hashtag
    """,
    )

    formatted_df = DiagnosisLoader().preprocess_diagnosis_columns(df=df.lazy())

    resulting_diagnoses = formatted_df.collect().get_column("value").to_list()
    assert set(resulting_diagnoses) == {"F431", "ALFC3", "F329"}
    types = formatted_df.collect().get_column("type").to_list()
    assert set(types) == {"A", "+", "B"}
