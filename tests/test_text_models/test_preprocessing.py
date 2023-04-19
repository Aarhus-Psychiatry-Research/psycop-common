import pandas as pd
from psycop_feature_generation.text_models.preprocessing import text_preprocessing


def test_text_preprocessing():
    df = pd.DataFrame({"value": ["Jeg er paf!", "Jeg hedder Erik, og er 97 år?"]})
    preprocessed_df = text_preprocessing(df=df)

    expected = pd.Series([" paf", " hedder erik"])

    assert pd.testing.assert_series_equal(preprocessed_df["value"], expected) is None
