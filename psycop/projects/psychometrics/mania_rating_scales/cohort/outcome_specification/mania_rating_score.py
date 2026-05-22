import pandas as pd

from psycop.projects.psychometrics.loaders.structured_sfi import mas_m


def get_mania_rating_scores(
    timestamps_only: bool = False, timestamp_as_value_col: bool = False
) -> pd.DataFrame:
    """
    Return all Mania rating scale observations with associated timestamps.

    Parameters
    ----------
    timestamps_only : bool, default=False
        Returns only the columns dw_ek_borger and timestamp if True.
    timestamp_as_value_col : bool, default=False
        If True, creates or overwrites the column value with the timestamp value.
        If False, returns the value of mania rating scale.

    Returns
    -------
    pd.DataFrame
        DataFrame containing at least:
        - dw_ek_borger
        - timestamp
        - value (Mania rating scale-score)

    """

    df = mas_m()

    if timestamp_as_value_col:
        df["value"] = df["timestamp"]
    else:
        df["value"] = df["value"]

    if timestamps_only:
        return df[["dw_ek_borger", "timestamp"]]

    return df[["dw_ek_borger", "timestamp", "value"]]


if __name__ == "__main__":
    df = get_mania_rating_scores()
