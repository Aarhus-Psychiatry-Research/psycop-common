"""Tests of value cleaner."""

import pandas as pd
from psycop_model_training.preprocessing.pre_split.processors.value_cleaner import (
    PreSplitValueCleaner,
)


def test_offset_so_no_negative_values():
    """Test that _offset_values_so_no_negative_values offset values, so min is 0."""

    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3],
            "outc_timestamp": [1, 2, 3, 4, 5],
            "pred_1": [-10, -5, 0, 5, 10],
            "pred_2": [-100, -50, 0, 50, 100],
            "pred_3": [0, 1, 2, 3, 4],
        },
    )

    expected_df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3],
            "outc_timestamp": [1, 2, 3, 4, 5],
            "pred_1": [0, 5, 10, 15, 20],
            "pred_2": [0, 50, 100, 150, 200],
            "pred_3": [0, 1, 2, 3, 4],
        },
    )

    df = PreSplitValueCleaner._offset_so_no_negative_values(df)

    for col in df.columns:
        pd.testing.assert_series_equal(df[col], expected_df[col])


def test_offset_so_no_negative_values_not_same_ordering():
    """Test that _offset_values_so_no_negative_values offset values, so min is 0."""

    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3],
            "outc_timestamp": [1, 2, 3, 4, 5],
            "pred_2": [-100, -50, 0, 50, 100],
            "pred_3": [0, 1, 2, 3, 4],
            "pred_1": [-10, -5, 0, 5, 10],
        },
    )

    expected_df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3],
            "outc_timestamp": [1, 2, 3, 4, 5],
            "pred_1": [0, 5, 10, 15, 20],
            "pred_2": [0, 50, 100, 150, 200],
            "pred_3": [0, 1, 2, 3, 4],
        },
    )

    df = PreSplitValueCleaner._offset_so_no_negative_values(df)

    for col in df.columns:
        pd.testing.assert_series_equal(df[col], expected_df[col])
