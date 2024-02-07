import pandas as pd

from psycop.common.feature_generation.text_models.preprocessing import text_preprocessing


def test_text_preprocessing():
    df = pd.DataFrame(
        {"value": ["Jeg er paf!!", "Jeg hedder Erik, og er 97 år?", "Hun ankommer ca. klokken 3"]}
    )
    preprocessed_df = text_preprocessing(df=df)

    expected_df = pd.DataFrame({"value": ["  paf", " hedder erik   97 ", " ankommer  klokken 3"]})

    assert pd.testing.assert_series_equal(preprocessed_df["value"], expected_df["value"]) is None
