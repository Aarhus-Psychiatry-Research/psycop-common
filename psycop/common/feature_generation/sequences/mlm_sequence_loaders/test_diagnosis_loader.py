from psycop.common.feature_generation.sequences.mlm_sequence_loaders.MLMLoaders import (
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
