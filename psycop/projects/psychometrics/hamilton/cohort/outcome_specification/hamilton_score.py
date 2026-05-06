import pandas as pd

from psycop.projects.psychometrics.loaders.structured_sfi import hamilton_d17


def get_hamilton_scores(
    timestamps_only: bool = False, timestamp_as_value_col: bool = False
) -> pd.DataFrame:
    """
    Return all Hamilton D17 observations with associated timestamps.

    Parameters
    ----------
    timestamps_only : bool, default=False
        Returns only the columns dw_ek_borger and timestamp if True.
    timestamp_as_value_col : bool, default=False
        If True, creates or overwrites the column value with the timestamp value.
        If False, sets value to 1 as a standard flag.

    Returns
    -------
    pd.DataFrame
        DataFrame containing at least:
        - dw_ek_borger
        - timestamp
        - ham_d17 (Hamilton D17 score)
        - value (optional)
    """

    df = hamilton_d17()

    if timestamp_as_value_col:
        df["value"] = df["timestamp"]
    else:
        df["value"] = df["value"]

    if timestamps_only:
        return df[["dw_ek_borger", "timestamp"]]

    return df[["dw_ek_borger", "timestamp", "value"]]


if __name__ == "__main__":
    df = get_hamilton_scores()
