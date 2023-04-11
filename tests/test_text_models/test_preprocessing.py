import pandas as pd
from psycop_feature_generation.text_models.preprocessing import (
    remove_stop_words_from_series,
    remove_symbols_from_series,
)


def test_remove_symbols_from_series():
    text_series = pd.Series(["Jeg er paf!", "jeg hedder, erik og er 97 år?"])
    text_series = remove_symbols_from_series(text_series=text_series)

    expected = pd.Series(["Jeg er paf ", "jeg hedder erik og er 97 år "])

    assert pd.testing.assert_series_equal(text_series, expected) is None


def test_remove_stop_words_from_series():
    text_series = pd.Series(["jeg er paf", "jeg hedder erik"])
    text_series = remove_stop_words_from_series(text_series=text_series)

    expected = pd.Series(["  paf", " hedder erik"])

    assert pd.testing.assert_series_equal(text_series, expected) is None
