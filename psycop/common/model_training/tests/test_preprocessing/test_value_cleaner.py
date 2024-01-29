"""Tests of value cleaner."""

import pandas as pd

from psycop.common.model_training.preprocessing.pre_split.processors.value_cleaner import (
    PreSplitValueCleaner,
)


def test_offset_so_no_negative_values():
    """Test that _offset_values_so_no_negative_values offset values, so min is 0."""

    df = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "outc_timestamp": [1, 2, 3],
            "pred_1": [-10, 0, 10],  # Monotonically increasing
            "pred_2": [100, -100, 0],  # If the first value is not min
            "pred_3": [0, 1, 2],  # No negative values, should not be offset
        }
    )

    expected_df = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "outc_timestamp": [1, 2, 3],
            "pred_1": [0, 10, 20],
            "pred_2": [200, 0, 100],
            "pred_3": [0, 1, 2],
        }
    )

    df = PreSplitValueCleaner._offset_so_no_negative_values(df)  # type: ignore

    for col in df.columns:
        pd.testing.assert_series_equal(df[col], expected_df[col])
