"""Test of unpack intervals to days"""

import pandas as pd

from psycop_feature_generation.loaders.raw.utils import (
    unpack_intervals_to_days,
)
from psycop_feature_generation.utils_for_testing import (
    str_to_df,
)


def test_unpack_intervals_to_days():
    df_str = """dw_ek_borger,datotid_start,datotid_slut,value
                1,2021-01-01 12:00:00,2021-01-01 13:00:00,1.0
                2,2021-02-02 00:00:00,2021-02-02 14:00:00,10.0
                3,2021-03-03 15:00:00,2021-03-05 15:30:00,48.5
                4,2021-04-04 00:00:00,2021-04-06 16:00:00,16.0
                5,2021-05-05 17:30:00,2021-05-07 00:00:00,30.5
                6,2021-06-06 00:00:00,2021-06-09 00:00:00,72.0
            """

    expected_df_str = """dw_ek_borger,timestamp,value
                1,2021-01-01 12:00:00,1
                1,2021-01-01 13:00:00,1
                2,2021-02-02 00:00:00,1
                2,2021-02-02 14:00:00,1
                3,2021-03-03 15:00:00,1
                3,2021-03-04 00:00:00,1
                3,2021-03-05 00:00:00,1
                3,2021-03-05 15:30:00,1
                4,2021-04-04 00:00:00,1
                4,2021-04-05 00:00:00,1
                4,2021-04-06 00:00:00,1
                4,2021-04-06 16:00:00,1
                5,2021-05-05 17:30:00,1
                5,2021-05-06 00:00:00,1
                5,2021-05-07 00:00:00,1
                6,2021-06-06 00:00:00,1
                6,2021-06-07 00:00:00,1
                6,2021-06-08 00:00:00,1
                6,2021-06-09 00:00:00,1
            """

    # 1: interval < 1 day and times are not 00:00:00 (= two rows with start and end time)
    # 2: interval < 1 day and start is 00:00:00 (= two rows with start and end time)
    # 3: interval > 1 day and times are not 00:00:00 (= one row with start time, rows with time 00:00:00 in-between, and one row with end time)
    # 4: interval > 1 day and start is 00:00:00 (= one row with start time 00:00:00, rows with time 00:00:00 in-between, and one row with end time)
    # 5: interval > 1 day and end is 00:00:00 (= one row with start time 00:00:00, rows with time 00:00:00 in-between, and one row with end time 00:00:00)
    # 6: interval > 1 day and both times are 00:00:00 (= one row per day, all times 00:00:00)

    df = str_to_df(df_str, convert_str_to_float=False)
    df["datotid_start"] = pd.to_datetime(df["datotid_start"])
    df["datotid_slut"] = pd.to_datetime(df["datotid_slut"])

    expected_df = str_to_df(expected_df_str, convert_str_to_float=False)
    expected_df["timestamp"] = pd.to_datetime(expected_df["timestamp"])

    df = unpack_intervals_to_days(
        df,
        starttime_col="datotid_start",
        endtime_col="datotid_slut",
    )

    for col in df.columns:
        pd.testing.assert_series_equal(df[col], expected_df[col])
