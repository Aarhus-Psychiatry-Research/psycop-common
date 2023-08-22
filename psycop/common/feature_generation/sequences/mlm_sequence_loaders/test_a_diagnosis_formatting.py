from psycop.common.feature_generation.sequences.mlm_sequence_loaders.diagnoses_sequences import (
    format_diagnosis_columns,
)
from psycop.common.test_utils.str_to_df import str_to_pl_df


def test_a_diagnosis_formatting():
    df = str_to_pl_df(
        """dw_ek_borger,datotid_slut,diagnosegruppestreng
    1,2023-01-01,A:DF431                    # Keep all, even though doesn't end with a hashtag 
    1,2023-01-01,A:DF431#+:ALFC3#B:DF329    # Keep up until, but not including, first hashtag 
    """
    )

    formatted_df = format_diagnosis_columns(df=df.lazy())

    resulting_diagnoses = formatted_df.collect().get_column("value").to_list()
    assert resulting_diagnoses == ["DF431", "DF431", "ALFC3", "DF329"]
