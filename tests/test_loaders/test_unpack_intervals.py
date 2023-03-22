"""Tests of unpack_intervals"""

import pandas as pd
from psycop_feature_generation.loaders.raw.utils import (
    unpack_intervals,
)
from psycop_feature_generation.utils_for_testing import (
    str_to_df,
)


def test_unpack_intervals_to_days():
    df_str = """dw_ek_borger,timestamp_start,timestamp_end,value
                1,2021-01-01 12:00:00,2021-01-01 13:00:00,1.0
                2,2021-03-03 15:00:00,2021-03-05 15:30:00,48.5
                3,2021-06-06 00:00:00,2021-06-09 00:00:00,72.0
            """

    expected_df_str = """dw_ek_borger,timestamp,value
                1,2021-01-01 12:00:00,1
                1,2021-01-01 13:00:00,1
                2,2021-03-03 15:00:00,1
                2,2021-03-04 15:00:00,1
                2,2021-03-05 15:00:00,1
                2,2021-03-05 15:30:00,1
                3,2021-06-06 00:00:00,1
                3,2021-06-07 00:00:00,1
                3,2021-06-08 00:00:00,1
                3,2021-06-09 00:00:00,1
            """

    # 1: interval < 1 day (= two rows, one with start time and one with end time)
    # 2: interval > 1 day and times are not 00:00:00 (= one row with start time, one row per day in-between with timestamp same as start time, and one row with end time)
    # 3: interval > 1 day and both times are 00:00:00 (= one row per day, includeing start and end day, all times 00:00:00)

    df = str_to_df(df_str, convert_str_to_float=False)
    expected_df = str_to_df(expected_df_str, convert_str_to_float=False)

    df = unpack_intervals(
        df,
        starttime_col="timestamp_start",
        endtime_col="timestamp_end",
        unpack_freq="D",
    )

    for col in df.columns:
        pd.testing.assert_series_equal(df[col], expected_df[col])


def test_unpack_intervals_to_5Hfreq():
    df_str = """dw_ek_borger,timestamp_start,timestamp_end,value
                1,2021-01-01 12:00:00,2021-01-01 13:00:00,1.0
                2,2021-02-02 15:00:00,2021-02-02 20:00:00,5.0
                3,2021-03-04 16:00:00,2021-03-05 4:00:00,12.0
            """

    expected_df_str = """dw_ek_borger,timestamp,value
                1,2021-01-01 12:00:00,1
                1,2021-01-01 13:00:00,1
                2,2021-02-02 15:00:00,1
                2,2021-02-02 20:00:00,1
                3,2021-03-04 16:00:00,1
                3,2021-03-04 21:00:00,1
                3,2021-03-05 02:00:00,1
                3,2021-03-05 4:00:00,1
            """

    # 1: interval < 5 hours
    # 2: interval = 5 hours
    # 3: interval > 5 hours

    df = str_to_df(df_str, convert_str_to_float=False)
    expected_df = str_to_df(expected_df_str, convert_str_to_float=False)

    df = unpack_intervals(
        df,
        starttime_col="timestamp_start",
        endtime_col="timestamp_end",
        unpack_freq="5H",
    )

    for col in df.columns:
        pd.testing.assert_series_equal(df[col], expected_df[col])
