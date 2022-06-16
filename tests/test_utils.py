from src.utils import difference_in_days, drop_records_if_datediff_days_smaller_than

from utils_for_testing import str_to_df


def test_difference_in_days():
    test_df = str_to_df(
        """timestamp_2,timestamp_1
    2021-01-01, 2021-01-31,
    2021-01-31, 2021-01-01,
    2021-01-01, 2021-01-01
    """
    )

    differences = difference_in_days(test_df["timestamp_2"], test_df["timestamp_1"])
    expected = [30.0, -30.0, 0.0]

    for i in range(len(differences)):
        assert differences[i] == expected[i]


def test_drop_records_if_datediff_days_smaller_than():
    test_df = str_to_df(
        """timestamp_2,timestamp_1
    2021-01-01, 2021-01-31,
    2021-01-31, 2021-01-01,
    2021-01-01, 2021-01-01
    """
    )

    drop_records_if_datediff_days_smaller_than(
        df=test_df,
        t2_col_name="timestamp_2",
        t1_col_name="timestamp_1",
        threshold_days=1,
        inplace=True,
    )

    differences = difference_in_days(test_df["timestamp_2"], test_df["timestamp_1"])
    expected = [30.0]

    for i in range(len(differences)):
        assert differences[i] == expected[i]
